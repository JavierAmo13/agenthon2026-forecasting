"""House Model access: text -> structured semantic JSON.

Calls $MODEL_ENDPOINT/v1/chat/completions (OpenAI-compatible) with
MODEL_NAME / MODEL_TOKEN from the environment, honouring the 25-request
per-unit budget. When the endpoint is absent (local dev), returns None and
the text pipeline degrades to zero semantic features — the numeric model
still runs.

Results are cached per document hash under .cache/llm/ so reruns do not
re-spend the budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import urllib.request

CACHE_DIR = pathlib.Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm"))

_SCHEMA = {
    "sentiment": "float in [-1,1]",
    "relevance": "float in [0,1]",
    "surprise": "float in [0,1]",
    "risk_intensity": "float in [0,1]",
    "direction": "hawkish|dovish|risk_on|risk_off|neutral",
    "event_type": "central_bank|macro_release|geopolitics|politics|other",
    "assets_mentioned": ["list of asset ids mentioned"],
}


def _endpoint() -> str | None:
    ep = os.environ.get("MODEL_ENDPOINT")
    return ep.rstrip("/") + "/v1" if ep else None


def available() -> bool:
    return bool(_endpoint() and os.environ.get("MODEL_NAME"))


def _cache_path(doc_id: str, text: str) -> pathlib.Path:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{doc_id}_{h}.json"


def extract_doc(doc_id: str, text: str, *, timeout: float = 60.0) -> dict | None:
    """One document -> semantic JSON (cached). None when House is unavailable."""
    cp = _cache_path(doc_id, text)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    if not available():
        return None
    prompt = (
        "Extract structured market information from this financial document. "
        "Return ONLY a JSON object with keys: " + json.dumps(_SCHEMA) +
        "\n\nDOCUMENT:\n" + text[:8000]
    )
    body = json.dumps({
        "model": os.environ["MODEL_NAME"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 400,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        _endpoint() + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('MODEL_TOKEN', '')}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        data = json.loads(content[content.index("{"):content.rindex("}") + 1])
    except Exception:
        return None
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # cache is best-effort; a read-only workdir must not break the run
    return data


def extract_corpus(texts, budget: int = 25) -> dict[str, dict]:
    """Per-doc semantic dicts, honoring the request budget (cache misses only)."""
    out: dict[str, dict] = {}
    spent = 0
    for doc in texts:
        cp = _cache_path(doc.doc_id, doc.text)
        if cp.exists():
            try:
                out[doc.doc_id] = json.loads(cp.read_text(encoding="utf-8"))
                continue
            except (ValueError, OSError):
                pass
        if spent >= budget:
            break
        res = extract_doc(doc.doc_id, doc.text)
        spent += 1
        if res:
            out[doc.doc_id] = res
    return out