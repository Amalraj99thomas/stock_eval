"""
Company Stock Evaluation API (FastAPI + RAG)
-------------------------------------------------
What this service does:
- Takes a company name, searches the web (Serper), scrapes pages (ScrapingFish),
  cleans them to Markdown (BeautifulSoup), summarizes with an LLM,
  and builds a lightweight vector index (Chroma) to enable Q&A (RAG).

Security note: DO NOT hardcode API keys. Use environment variables.
Required env vars: SERPER_API_KEY, SCRAPINGFISH_API_KEY, OPENAI_API_KEY
Optional env vars: EMBED_MODEL (defaults to text-embedding-3-large), CHAT_MODEL (defaults to gpt-4o-mini)
"""
from __future__ import annotations

# -------------------- imports --------------------
import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -------------------- App setup --------------------
app = FastAPI(title="Company Stock Evaluation", version="1.0")
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("stock_evals")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env if present (you can also run: uvicorn main:app --reload --env-file .env)
try:
    load_dotenv(find_dotenv(), override=False)
except Exception:
# Non-fatal if dotenv isn't installed or file not found
    pass

# Models (override via env if needed)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# -------------------- Environment keys --------------------
#--------------------environment keys------------


SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SCRAPING_API_KEY = os.getenv("SCRAPINGFISH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SERPER_API_KEY:
    logging.warning("SERPER_API_KEY not set; /name will fail until configured.")
if not SCRAPING_API_KEY:
    logging.warning("SCRAPINGFISH_API_KEY not set; scraping will fallback to direct GET where allowed.")
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set; LLM calls will fail until configured.")

# -------------------- Pydantic Schemas --------------------
class ResultRelevance(BaseModel):
    explanation: str
    id: str

class RelevanceCheckOutput(BaseModel):
    relevant_results: List[ResultRelevance]

class NameIn(BaseModel):
    company_name: str = Field(..., description="Company name, e.g., 'NVIDIA' or 'Apple' or 'Tata Motors'")

class NameOut(BaseModel):
    doc_id: str
    company_name: str
    sources_kept: int
    message: str

class AskIn(BaseModel):
    doc_id: str = Field(..., description="The slug returned by /name for this company")
    question: str = Field(..., description="Your question about the company's shares/performance")

class AskOut(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

# -------------------- Helpers --------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def file_safe_name(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- LLM + Embeddings --------------------

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)


def get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

# -------------------- Serper Search --------------------

def serper_search(queries: List[str]) -> Dict[str, Any]:
    """Call Serper API for a list of queries and merge unique results."""
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    all_results: Dict[str, Dict[str, Any]] = {}

    for q in queries:
        payload = {"q": q}
        logging.info(f"Serper search: {q}")
        resp = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            logging.warning(f"Serper error {resp.status_code}: {resp.text[:200]}")
            continue
        data = resp.json()
        # Serper returns multiple sections; we prioritise 'organic' and 'news'
        for section in ("news", "organic"):
            for item in data.get(section, []) or []:
                link = item.get("link")
                if not link:
                    continue
                key = file_safe_name(link)
                all_results[key] = {
                    "id": key,
                    "title": item.get("title") or "",
                    "snippet": item.get("snippet") or item.get("snippet_highlighted", ""),
                    "link": link,
                    "source": section,
                }
    return {"results": list(all_results.values())}

# -------------------- Relevance Filtering via LLM --------------------

def relevant_results_prompt(company: str, items: List[Dict[str, Any]]) -> ChatPromptTemplate:
    numbered = []
    for idx, it in enumerate(items, start=1):
        numbered.append(
            f"[{idx}]\nTitle: {it.get('title','')}\nURL: {it.get('link','')}\nSnippet: {it.get('snippet','')}\n"
        )
    listing = "\n\n".join(numbered)

    system = (
        "You are a precise research assistant. Given a list of web results, "
        "select up to 8 that best help evaluate *current* stock performance, price, "
        "recent earnings, analyst sentiment, and market news for the target company. "
        "Prefer high-quality finance sources (e.g., exchange/Nasdaq, investor relations, Yahoo Finance, Bloomberg, CNBC). "
        "Return STRICT JSON with key 'relevant_results' as an array of objects {id, explanation}. "
        "Use the given numeric index as 'id'."
    )
    user = (
        f"Company: {company}\n\nResults:\n{listing}\n\n"
        "Return JSON only, no commentary."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user),
    ])


def check_relevance(company: str, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = search_results.get("results", [])
    if not items:
        return []

    index_to_item = {str(i + 1): it for i, it in enumerate(items)}
    prompt = relevant_results_prompt(company, items)

    llm_structured = get_llm().with_structured_output(RelevanceCheckOutput)
    try:
        # chain: prompt -> structured LLM
        data: RelevanceCheckOutput = (prompt | llm_structured).invoke({})
        chosen = []
        for rr in data.relevant_results[:8]:
            idx = str(rr.id)
            if idx in index_to_item:
                it = index_to_item[idx].copy()
                it["why"] = rr.explanation
                chosen.append(it)
        if chosen:
            return chosen
    except Exception as e:
        logging.warning(f"check_relevance LLM parse failed (structured): {e}")

    # fallback heuristic
    preferred = ["nasdaq.com","nyse.com","investor","ir.","finance.yahoo.com",
                 "bloomberg.com","cnbc.com","reuters.com","ft.com","seekingalpha.com"]
    def score(u: str) -> int: return sum(1 for p in preferred if p in (u or "").lower())
    return sorted(items, key=lambda it: score(it.get("link")), reverse=True)[:6]
# -------------------- Scraping + HTML -> Markdown --------------------

def scrapingfish_get(url: str, render_js: bool = False) -> Optional[str]:
    if not SCRAPING_API_KEY:
        return None
    params = {"api_key": SCRAPING_API_KEY, "url": url, "render_js": "true" if render_js else "false"}
    base = os.getenv("SCRAPINGFISH_BASE", "https://api.scrapingfish.com/v1")
    for attempt in range(2):
        try:
            resp = requests.get(base, params=params, timeout=45)
            if resp.status_code == 200:
                ctype = resp.headers.get("content-type", "")
                if "application/json" in ctype:
                    try:
                        js = resp.json()
                        return js.get("html") or js.get("content") or js.get("data")
                    except Exception:
                        pass
                return resp.text
            logging.warning(f"ScrapingFish {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logging.warning(f"ScrapingFish error: {e}")
        time.sleep(0.5)
    return None


def direct_get(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code == 200:
            ctype = resp.headers.get("content-type", "")
            if "application/pdf" in ctype:
                logging.info(f"Skip PDF (not parsed here): {url}")
                return None
            if "text/html" in ctype:
                return resp.text
    except Exception as e:
        logging.warning(f"Direct GET failed: {e}")
    return None


def html_to_markdown(html: str, base_url: str) -> str:
    """Very lightweight HTML -> Markdown using BeautifulSoup.
    Not perfect but good enough for Q&A context building.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles/navs
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "img"]):
        tag.decompose()

    # Convert links to [text](href)
    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        href = a.get("href")
        if href and text:
            a.string = f"[{text}]({href})"
        elif text:
            a.string = text

    # Headings -> markdown
    out_lines: List[str] = []
    for el in soup.body.find_all(recursive=False) if soup.body else soup.find_all(recursive=False):
        out_lines.extend(_element_to_md_lines(el))

    text = "\n".join(line for line in out_lines if line is not None)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Include provenance footer
    return text + f"\n\n---\nSource: {base_url}\n"


def _element_to_md_lines(el) -> List[str]:
    lines: List[str] = []
    name = getattr(el, "name", None)
    if not name:
        val = el.string.strip() if isinstance(el.string, str) else el.get_text(" ", strip=True)
        if val:
            return [val]
        return []

    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        text = el.get_text(" ", strip=True)
        lines.append("#" * level + " " + text)
    elif name == "p":
        txt = el.get_text(" ", strip=True)
        if txt:
            lines.append(txt)
    elif name in {"ul", "ol"}:
        ordered = name == "ol"
        idx = 1
        for li in el.find_all("li", recursive=False):
            t = li.get_text(" ", strip=True)
            if t:
                bullet = f"{idx}. " if ordered else "- "
                lines.append(bullet + t)
                idx += 1
    else:
        for child in el.find_all(recursive=False):
            lines.extend(_element_to_md_lines(child))
    return lines

# -------------------- Summarization Prompts --------------------

def markdown_summary_prompt(company: str) -> ChatPromptTemplate:
    system = (
        "You are an equity research assistant. Summarize the markdown into crisp, factual bullets. "
        "Focus on current stock price/performance, recent earnings, guidance, analyst/ratings, "
        "major news, risks, and any quantitative figures (with units). If dates are present, keep them. "
        "If off-topic for the target company, say 'Irrelevant'."
    )
    user = (
        "Company: {company}\n\nMarkdown:\n```\n{markdown}\n```\n\n"
        "Return 5-12 bullets labeled '- ' and finish with 'Source: {source}'."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user),
    ])


def final_summary_prompt(company: str) -> ChatPromptTemplate:
    system = (
        "You are a senior equity analyst. Merge multiple source summaries into a concise briefing. "
        "Resolve small conflicts by preferring more recent or primary sources (exchanges/IR). "
        "Do NOT fabricate numbers. If something is unclear, note the uncertainty."
    )
    user = (
        "Company: {company}\n\nSummaries from sources:\n```\n{summaries}\n```\n\n"
        "Produce: (1) 6-12 key takeaways; (2) 2-3 watch items; (3) A 2-sentence quick answer to: "
        "'How are the company's shares doing right now?'"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user),
    ])

# -------------------- Pipeline Steps --------------------

def scrape_save_markdown(company_dir: Path, chosen: List[Dict[str, Any]]) -> List[Path]:
    md_dir = company_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for it in chosen:
        url = it.get("link")
        if not url:
            continue
        html = scrapingfish_get(url) or direct_get(url)
        if not html:
            logging.info(f"Skip (no html): {url}")
            continue
        md = html_to_markdown(html, url)
        fname = f"{file_safe_name(url)}.md"
        path = md_dir / fname
        path.write_text(md, encoding="utf-8")
        saved_paths.append(path)
        time.sleep(0.6)  # politeness
    return saved_paths


def generate_markdown_summaries(company: str, md_files: List[Path]) -> Tuple[str, List[Document]]:
    llm = get_llm()
    prompt_t = markdown_summary_prompt(company)
    summaries: List[str] = []
    docs: List[Document] = []

    for p in md_files:
        md_text = p.read_text(encoding="utf-8")
        # Truncate overly long inputs for cost-control; split if needed
        if len(md_text) > 18000:
            md_text = md_text[:18000]
        msg = llm.invoke(prompt_t.format(company=company, markdown=md_text, source=p.name))
        summary = msg.content.strip()
        # Filter out irrelevant
        if summary.lower().startswith("irrelevant"):
            continue
        summaries.append(summary)
        # Document per-summary for vector store, with source metadata
        src_url = _extract_source_from_summary(summary) or p.name
        docs.append(Document(page_content=summary, metadata={"source": src_url}))
        time.sleep(0.3)

    combined = "\n\n".join(summaries)
    return combined, docs


def _extract_source_from_summary(summary: str) -> Optional[str]:
    m = re.search(r"Source:\s*(.+)$", summary, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def generate_final_summary(company: str, summaries_blob: str) -> str:
    if not summaries_blob.strip():
        return ""
    llm = get_llm()
    prompt_t = final_summary_prompt(company)
    msg = llm.invoke(prompt_t.format(company=company, summaries=summaries_blob))
    return msg.content.strip()


def build_vectorstore(company_dir: Path, docs: List[Document], final_summary: str) -> Chroma:
    # Add the final summary as its own doc to improve general answers
    if final_summary:
        docs = docs + [Document(page_content=final_summary, metadata={"source": "final_summary"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks: List[Document] = splitter.split_documents(docs)

    persist_dir = company_dir / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedder(),
        persist_directory=str(persist_dir),
        collection_name="stock_eval_docs",
    )
    # vectordb.persist()
    return vectordb


def load_vectorstore(company_dir: Path) -> Optional[Chroma]:
    persist_dir = company_dir / "chroma"
    if not persist_dir.exists():
        return None
    try:
        return Chroma(
            embedding_function=get_embedder(),
            persist_directory=str(persist_dir),
            collection_name="stock_eval_docs",
        )
    except Exception as e:
        logging.warning(f"Failed to load Chroma: {e}")
        return None


def build_qa_chain(vectordb: Chroma):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
    llm = get_llm()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You answer questions about a company's stock using the provided context. "
         "Cite sources inline like [Source: <value>] when you use them. If unsure, say you don't know."),
        # IMPORTANT: expect {input}, not {question}
        ("user", "Question: {input}\n\nContext:\n{context}")
    ])

    combine_chain = create_stuff_documents_chain(llm, qa_prompt)
    # No custom input_key/context_key — use defaults: input_key="input", context_key="context"
    return create_retrieval_chain(retriever, combine_chain)

# -------------------- Routes --------------------
@app.post("/name", response_model=NameOut)
async def register_company(payload: NameIn):
    """Orchestrates the pipeline for a given company name and builds its index."""
    company = payload.company_name.strip()
    if not company:
        raise HTTPException(status_code=400, detail="Company name cannot be empty.")

    doc_id = slugify(company)
    company_dir = DATA_DIR / doc_id
    company_dir.mkdir(parents=True, exist_ok=True)

    # 1) Search queries
    queries = [
        f"{company} stock price",
        f"{company} latest earnings",
        f"{company} site:nasdaq.com OR site:nyse.com",
        f"{company} site:finance.yahoo.com",
        f"{company} site:cnbc.com OR site:bloomberg.com",
    ]

    if not SERPER_API_KEY:
        raise HTTPException(status_code=500, detail="SERPER_API_KEY missing.")

    search = serper_search(queries)
    save_json(company_dir / "search.json", search)

    # 2) Relevance filtering with LLM
    chosen = check_relevance(company, search)
    if not chosen:
        raise HTTPException(status_code=404, detail="No relevant results found.")
    save_json(company_dir / "relevant.json", {"chosen": chosen})

    # 3) Scrape + Markdown
    md_files = scrape_save_markdown(company_dir, chosen)
    if not md_files:
        raise HTTPException(status_code=502, detail="Failed to scrape any relevant pages.")

    # 4) Summaries per-source
    summaries_blob, summary_docs = generate_markdown_summaries(company, md_files)
    (company_dir / "summaries.txt").write_text(summaries_blob, encoding="utf-8")

    # 5) Final summary
    final_sum = generate_final_summary(company, summaries_blob)
    (company_dir / "final_summary.md").write_text(final_sum, encoding="utf-8")

    # 6) Vectorstore
    vectordb = build_vectorstore(company_dir, summary_docs, final_sum)
    _ = vectordb  # ensure built

    return NameOut(
        doc_id=doc_id,
        company_name=company,
        sources_kept=len(md_files),
        message="Index built. Use /ask with this doc_id to query.",
    )


@app.post("/ask", response_model=AskOut)
async def ask(payload: AskIn):
    company_dir = DATA_DIR / payload.doc_id
    if not company_dir.exists():
        raise HTTPException(status_code=404, detail="doc_id not found. Run /name first.")

    vectordb = load_vectorstore(company_dir)
    if not vectordb:
        raise HTTPException(status_code=500, detail="Vector store missing or failed to load.")

    chain = build_qa_chain(vectordb)
    result = chain.invoke({"input": payload.question})

    # Extract sources from the context documents if available
    docs = result.get("context") or result.get("input_documents") or result.get("documents") or []
    sources = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source")
        if src and src not in sources:
            sources.append(src)
            
    # Normalize to list of dicts
    source_dicts = [{"source": s} for s in sources]

    return AskOut(answer=result.get("answer") or result.get("output") or "", sources=source_dicts)


# -------------------- Tiny index page --------------------
@app.get("/")
def root():
    return {
        "name": "Company Stock Evaluation API",
        "routes": {
            "POST /name": "Build RAG index for a company (body: {company_name})",
            "POST /ask": "Ask a question with {doc_id, question}",
        },
        "data_dir": str(DATA_DIR),
    }
