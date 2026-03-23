import asyncio
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, EmailStr

pipeline = None
evaluator = None
vs = None
emb = None
user_histories: dict[str, dict] = {}
app_settings = {"preferred_provider": "auto"}

HISTORY_DIR = Path("data/history")
_ANON_USER = "__anon__"


def _history_path(user_id: str) -> Path:
    safe = user_id.replace("/", "_").replace("\\", "_")
    return HISTORY_DIR / f"{safe}.json"


def _get_user_history(user_id: str) -> dict:
    if user_id in user_histories:
        return user_histories[user_id]
    path = _history_path(user_id)
    data = {"history": [], "fact_checks": 0}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    user_histories[user_id] = data
    return data


def _save_user_history(user_id: str):
    data = user_histories.get(user_id, {"history": [], "fact_checks": 0})
    data["history"] = data["history"][-100:]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _history_path(user_id).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _uid(user: dict | None) -> str:
    return (user or {}).get("id") or _ANON_USER


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, evaluator, vs, emb
    from core.embedder import Embedder
    from core.vectorstore import VectorStore
    from core.retriever import Retriever
    from core.reasoner import Reasoner
    from core.evaluator import Evaluator
    from core.calibrator import Calibrator
    from core.pipeline import Pipeline
    from utils.wiki import WikiFetcher
    from core.auth import init_db, close_db

    auth_ok = await init_db()
    print(f"[VerifAI] Auth: {'MongoDB connected' if auth_ok else 'disabled (no MONGODB_URI)'}")

    emb = Embedder()
    vs = VectorStore(embedder=emb)
    migrated = vs.migrate_untagged_chunks()
    if migrated:
        print(f"[VerifAI] Migrated {migrated} chunks → tagged as __system__")
    evaluator = Evaluator(emb)
    pipeline = Pipeline(
        emb, vs, Retriever(vs, emb), Reasoner(),
        evaluator, Calibrator(), WikiFetcher(),
    )
    print(f"[VerifAI] Ready — KB has {vs.count()} chunks")
    _old_hist = Path("data/history.json")
    if _old_hist.exists():
        try:
            _old = json.loads(_old_hist.read_text(encoding="utf-8"))
            anon_data = _get_user_history(_ANON_USER)
            anon_data["history"] = _old.get("history", [])
            anon_data["fact_checks"] = _old.get("fact_checks", 0)
            _save_user_history(_ANON_USER)
            _old_hist.rename(_old_hist.with_suffix(".json.bak"))
            print("[VerifAI] Migrated old history.json → per-user format")
        except Exception:
            pass
    print(f"[VerifAI] Per-user history enabled (dir: {HISTORY_DIR})")
    yield
    for uid in user_histories:
        _save_user_history(uid)
    await close_db()


app = FastAPI(title="VerifAI", lifespan=lifespan)


class VerifyRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SettingsRequest(BaseModel):
    preferred_provider: str = "auto"
    verified_threshold: float | None = None
    contradicted_threshold: float | None = None
    confidence_threshold: float | None = None
    max_cycles: int | None = None

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def _result_dict(r):
    return {
        "query": r.query,
        "answer": r.answer or "",
        "model_used": r.model_used or "",
        "fallback_level": r.fallback_level,
        "semantic_consistency": round(r.semantic_consistency, 4),
        "factual_correctness": round(r.factual_correctness, 4),
        "confidence": round(r.confidence, 4),
        "sources": r.sources,
        "total_sources_used": r.total_sources_used,
        "calibration_cycles": r.calibration_cycles,
        "total_time_ms": round(r.total_time_ms, 1),
        "error": r.error,
        "trace": [
            {"step_type": s.step_type.value, "cycle": s.cycle,
             "detail": s.detail, "duration_ms": round(s.duration_ms, 1), "data": s.data or {}}
            for s in (r.trace or [])
        ],
    }


async def get_current_user(request: Request) -> dict | None:
    from core.auth import is_auth_enabled, decode_token

    if not is_auth_enabled():
        return None

    token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = request.cookies.get("verifai_token")

    if not token:
        raise HTTPException(401, "Not authenticated")

    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")

    return {"id": payload["sub"], "email": payload["email"], "name": payload["name"]}


@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    from core.auth import is_auth_enabled, create_user, create_token
    if not is_auth_enabled():
        raise HTTPException(503, "Auth not configured")

    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if len(req.name.strip()) < 1:
        raise HTTPException(400, "Name is required")

    try:
        user = await create_user(req.name, req.email, req.password)
    except ValueError as e:
        raise HTTPException(409, str(e))

    token = create_token(user["id"], user["email"], user["name"])
    return {"token": token, "user": user}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    from core.auth import is_auth_enabled, authenticate_user, create_token
    if not is_auth_enabled():
        raise HTTPException(503, "Auth not configured")

    user = await authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")

    token = create_token(user["id"], user["email"], user["name"])
    return {"token": token, "user": user}


@app.get("/api/auth/me")
async def auth_me(user=Depends(get_current_user)):
    from core.auth import is_auth_enabled
    if not is_auth_enabled():
        return {"auth_enabled": False}
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"auth_enabled": True, "user": user}


@app.get("/api/auth/status")
async def auth_status():
    from core.auth import is_auth_enabled
    return {"auth_enabled": is_auth_enabled()}


@app.get("/api/stats")
def get_stats(user=Depends(get_current_user)):
    uid = _uid(user)
    stats = vs.stats(uid)
    hist_data = _get_user_history(uid)
    query_history = hist_data["history"]
    return {
        "kb": stats,
        "queries": len(query_history),
        "avg_confidence": (
            round(sum(q["confidence"] for q in query_history) / len(query_history), 3)
            if query_history else 0
        ),
        "fact_checks": hist_data["fact_checks"],
        "history": query_history[-20:],
    }


@app.get("/api/config")
def get_config(user=Depends(get_current_user)):
    from core.config import FACT_CHECK_EXAMPLES, XRAY_QUERIES, EXAMPLE_QUERIES, verifier
    return {
        "qa_examples": EXAMPLE_QUERIES,
        "verify_examples": FACT_CHECK_EXAMPLES,
        "xray_examples": XRAY_QUERIES,
        "thresholds": {
            "verified": verifier.verified_threshold,
            "contradicted": verifier.contradicted_threshold,
        },
    }


@app.get("/api/settings")
def get_settings(user=Depends(get_current_user)):
    from core.config import api_keys, verifier, calibration
    uid = _uid(user)
    return {
        "preferred_provider": app_settings["preferred_provider"],
        "verified_threshold": verifier.verified_threshold,
        "contradicted_threshold": verifier.contradicted_threshold,
        "confidence_threshold": calibration.confidence_threshold,
        "max_cycles": calibration.max_cycles,
        "available": {
            "gemini": api_keys.has_gemini(),
            "groq": api_keys.has_groq(),
            "kb_chunks": vs.count(uid) if vs else 0,
        },
    }


@app.post("/api/settings")
def update_settings(req: SettingsRequest, user=Depends(get_current_user)):
    if req.preferred_provider not in ("auto", "gemini", "groq_primary", "groq_fallback"):
        raise HTTPException(400, "Invalid provider")
    app_settings["preferred_provider"] = req.preferred_provider
    import core.llm as llm_module
    llm_module.preferred_provider = req.preferred_provider

    from core.config import verifier, calibration
    changes = []
    if req.verified_threshold is not None and 0.5 <= req.verified_threshold <= 0.95:
        verifier.verified_threshold = round(req.verified_threshold, 2)
        changes.append(f"Verified threshold → {verifier.verified_threshold}")
    if req.contradicted_threshold is not None and 0.05 <= req.contradicted_threshold <= 0.5:
        verifier.contradicted_threshold = round(req.contradicted_threshold, 2)
        changes.append(f"Contradicted threshold → {verifier.contradicted_threshold}")
    if req.confidence_threshold is not None and 0.3 <= req.confidence_threshold <= 0.95:
        calibration.confidence_threshold = round(req.confidence_threshold, 2)
        changes.append(f"Confidence threshold τ → {calibration.confidence_threshold}")
    if req.max_cycles is not None and 1 <= req.max_cycles <= 5:
        calibration.max_cycles = req.max_cycles
        changes.append(f"Max calibration cycles → {calibration.max_cycles}")

    return {
        "preferred_provider": app_settings["preferred_provider"],
        "verified_threshold": verifier.verified_threshold,
        "contradicted_threshold": verifier.contradicted_threshold,
        "confidence_threshold": calibration.confidence_threshold,
        "max_cycles": calibration.max_cycles,
        "changes": changes,
    }


@app.get("/api/qa")
async def qa_stream(query: str, request: Request, token: str | None = None):
    from core.auth import is_auth_enabled, decode_token as _decode
    user_payload = None
    if is_auth_enabled():
        jwt_token = token or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        user_payload = _decode(jwt_token) if jwt_token else None
        if not user_payload:
            raise HTTPException(401, "Not authenticated")
    if not query or not query.strip():
        raise HTTPException(400, "Query required")

    uid = (user_payload or {}).get("sub") or _ANON_USER

    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        import core.llm as llm_module
        llm_module.preferred_provider = app_settings["preferred_provider"]

        def on_step(step):
            loop.call_soon_threadsafe(queue.put_nowait, step)

        async def run_pipeline():
            result = await loop.run_in_executor(
                None, lambda: pipeline.run(query.strip(), on_step=on_step, user_id=uid)
            )
            loop.call_soon_threadsafe(queue.put_nowait, result)

        task = asyncio.create_task(run_pipeline())

        while True:
            item = await queue.get()
            if hasattr(item, 'answer'):
                d = _result_dict(item)
                hist_data = _get_user_history(uid)
                hist_data["history"].append({
                    "query": item.query,
                    "confidence": item.confidence,
                    "model": item.model_used,
                    "s_c": round(item.semantic_consistency, 4),
                    "f_c": round(item.factual_correctness, 4),
                    "time_ms": round(item.total_time_ms, 1),
                    "cycles": item.calibration_cycles,
                    "sources": item.total_sources_used,
                })
                _save_user_history(uid)
                yield f"data: {json.dumps({'type': 'result', **d})}\n\n"
                yield "data: [DONE]\n\n"
                break
            else:
                payload = json.dumps({
                    "type": "step",
                    "step_type": item.step_type.value,
                    "cycle": item.cycle,
                    "detail": item.detail,
                    "duration_ms": round(item.duration_ms, 1),
                    "data": item.data or {},
                })
                yield f"data: {payload}\n\n"

        await task

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/verify-stream")
async def verify_stream(text: str, request: Request, token: str | None = None):
    from core.auth import is_auth_enabled, decode_token as _decode
    user_payload = None
    if is_auth_enabled():
        jwt_token = token or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        user_payload = _decode(jwt_token) if jwt_token else None
        if not user_payload:
            raise HTTPException(401, "Not authenticated")
    if not text or not text.strip():
        raise HTTPException(400, "Text required")

    uid = (user_payload or {}).get("sub") or _ANON_USER
    import time as _time
    from core.evaluator import extract_claims, nli_batch_scores, ClaimVerdict
    from core.config import verifier

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        t_total = _time.time()

        t0 = _time.time()
        claims = extract_claims(text)[:6]
        t_extract = round((_time.time() - t0) * 1000)
        step1 = {"step": "Extract Claims", "icon": "scissors", "time_ms": t_extract,
                  "detail": f"Split input into {len(claims)} atomic claims"}
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "step", "index": 0, **step1})

        t0 = _time.time()
        evidence = vs.query(text[:500], k=5, user_id=uid)
        t_retrieve = round((_time.time() - t0) * 1000)
        if not evidence:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": "Knowledge Base is empty. Add documents first."})
            return
        sources = list({(c.doc_name if hasattr(c, 'doc_name') else c.get('metadata', {}).get('doc_name', '?')) for c in evidence})

        RELEVANCE_FLOOR = 0.25
        best_sim = max((c.get('similarity', 0) if isinstance(c, dict) else 0) for c in evidence)
        off_topic = best_sim < RELEVANCE_FLOOR

        if off_topic:
            step2 = {"step": "Retrieve Evidence", "icon": "database", "time_ms": t_retrieve,
                      "detail": f"No relevant evidence found (best similarity: {best_sim:.2f})"}
        else:
            step2 = {"step": "Retrieve Evidence", "icon": "database", "time_ms": t_retrieve,
                      "detail": f"Found {len(evidence)} chunks from {len(sources)} source(s)"}
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "step", "index": 1, **step2})

        t0 = _time.time()

        if off_topic:
            verdicts = [
                ClaimVerdict(claim=claim, verdict='UNVERIFIABLE', nli_score=0.0,
                             evidence='', evidence_source='')
                for claim in claims
            ]
            t_nli = round((_time.time() - t0) * 1000)
            step3 = {"step": "Entailment Scoring", "icon": "brain", "time_ms": t_nli,
                      "detail": "Skipped — input is outside knowledge base domain"}
        else:
            top_chunks = evidence[:3]
            pairs = []
            pair_map = []
            for ci, claim in enumerate(claims):
                for chi, chunk in enumerate(top_chunks):
                    ct = chunk.text if hasattr(chunk, 'text') else chunk.get('text', '')
                    pairs.append((ct[:400], claim))
                    pair_map.append((ci, chi))

            scores = nli_batch_scores(pairs, max_length=512)

            verdicts = []
            for ci, claim in enumerate(claims):
                best_score = 0.0
                best_evidence = ''
                best_source = ''
                for j, (c_idx, ch_idx) in enumerate(pair_map):
                    if c_idx == ci and scores[j] > best_score:
                        best_score = scores[j]
                        chunk = top_chunks[ch_idx]
                        best_evidence = (chunk.text if hasattr(chunk, 'text') else chunk.get('text', ''))[:300]
                        best_source = chunk.doc_name if hasattr(chunk, 'doc_name') else chunk.get('metadata', {}).get('doc_name', 'unknown')
                if best_score < verifier.contradicted_threshold:
                    best_evidence = ''
                    best_source = ''
                if best_score >= verifier.verified_threshold:
                    verdict_str = 'VERIFIED'
                elif best_score <= verifier.contradicted_threshold:
                    verdict_str = 'CONTRADICTED'
                else:
                    verdict_str = 'UNVERIFIABLE'
                verdicts.append(ClaimVerdict(
                    claim=claim, verdict=verdict_str,
                    nli_score=round(best_score, 4),
                    evidence=best_evidence, evidence_source=best_source,
                ))

            t_nli = round((_time.time() - t0) * 1000)
            step3 = {"step": "Entailment Scoring", "icon": "brain", "time_ms": t_nli,
                      "detail": f"Scored {len(pairs)} claim-evidence pairs via entailment engine"}
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "step", "index": 2, **step3})

        t0 = _time.time()
        v_count = sum(1 for v in verdicts if v.verdict == "VERIFIED")
        c_count = sum(1 for v in verdicts if v.verdict == "CONTRADICTED")
        u_count = sum(1 for v in verdicts if v.verdict == "UNVERIFIABLE")
        t_verdict = round((_time.time() - t0) * 1000)
        step4 = {"step": "Verdict Classification", "icon": "scale", "time_ms": t_verdict,
                  "detail": f"Thresholds: verified ≥ {verifier.verified_threshold}, contradicted ≤ {verifier.contradicted_threshold}"}
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "step", "index": 3, **step4})

        total_ms = round((_time.time() - t_total) * 1000)

        loop.call_soon_threadsafe(queue.put_nowait, {
            "type": "result",
            "claims": [
                {"claim": v.claim, "verdict": v.verdict,
                 "nli_score": round(v.nli_score, 4),
                 "evidence": v.evidence or "", "evidence_source": v.evidence_source or ""}
                for v in verdicts
            ],
            "thresholds": {"verified": verifier.verified_threshold, "contradicted": verifier.contradicted_threshold},
            "total_ms": total_ms,
        })

    async def generate():
        future = loop.run_in_executor(None, _run)
        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("type") in ("result", "error"):
                break
        await future
        hist_data = _get_user_history(uid)
        hist_data["fact_checks"] = hist_data.get("fact_checks", 0) + 1
        _save_user_history(uid)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/xray")
async def xray(query: str, user=Depends(get_current_user)):
    if not query or not query.strip():
        raise HTTPException(400, "Query required")
    uid = _uid(user)
    loop = asyncio.get_event_loop()
    q = query.strip()
    r_full, r_basic, r_llm = await asyncio.gather(
        loop.run_in_executor(None, lambda: pipeline.run(q, user_id=uid)),
        loop.run_in_executor(None, lambda: pipeline.run_basic_rag(q, user_id=uid)),
        loop.run_in_executor(None, lambda: pipeline.run_llm_only(q)),
    )
    return {
        "autorag": _result_dict(r_full),
        "basic_rag": _result_dict(r_basic),
        "llm_only": _result_dict(r_llm),
    }


@app.get("/api/kb/documents")
def kb_documents(user=Depends(get_current_user)):
    uid = _uid(user)
    return vs.list_documents(uid)


@app.delete("/api/kb/documents/{doc_name:path}")
def kb_delete(doc_name: str, user=Depends(get_current_user)):
    uid = _uid(user)
    deleted = vs.delete_document(doc_name, user_id=uid)
    return {"deleted": deleted}


@app.post("/api/kb/upload")
async def kb_upload(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    uid = _uid(user)
    doc_name = file.filename.rsplit(".", 1)[0]
    if vs.has_document(doc_name, user_id=uid):
        raise HTTPException(409, f"'{doc_name}' already exists. Delete it first.")

    from utils.pdf_loader import load_pdf_bytes
    content = await file.read()
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(
        None, lambda: load_pdf_bytes(content, doc_name=doc_name, source="uploaded")
    )
    if not chunks:
        raise HTTPException(422, "No text could be extracted from this PDF")

    added = vs.add_chunks(chunks, user_id=uid)
    return {"doc_name": doc_name, "chunks_added": added}


@app.post("/api/kb/search")
async def kb_search(req: SearchRequest, user=Depends(get_current_user)):
    uid = _uid(user)
    return vs.query(req.query, k=req.k, user_id=uid)


class SeedRequest(BaseModel):
    topic: str = ""

@app.post("/api/kb/seed")
async def kb_seed(req: SeedRequest | None = None, user=Depends(get_current_user)):
    from utils.wiki import WikiFetcher
    uid = _uid(user)
    topic = (req.topic.strip() if req and req.topic else "") or None
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: WikiFetcher().smart_seed(vs, topic=topic, user_id=uid)
    )
    return {"chunks_added": result["total"], "wiki": result["wiki"], "web": result["web"]}


_static = Path(__file__).parent / "static"


@app.get("/")
def serve_index():
    return FileResponse(str(_static / "index.html"))


@app.get("/favicon.svg")
def serve_favicon():
    return FileResponse(str(_static / "favicon.svg"), media_type="image/svg+xml")


@app.get("/login")
def serve_login():
    return FileResponse(str(_static / "login.html"))


app.mount("/css", StaticFiles(directory=str(_static / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(_static / "js")), name="js")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=5050, reload=False)
