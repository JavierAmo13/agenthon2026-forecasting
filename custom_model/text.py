
"""House model client + text->adjustment extraction.

One call per unit (the budget allows 25; a single well-built call beats a
chatty loop — the official reasoning baseline makes one call and never
retries). The model reads dated documents, returns bounded JSON, and the
caller applies the adjustments with hard clamps; everything is logged to
forecast_meta.json and forecast_rationale.md so a reviewer can audit which
document drove which number.

The prompt asks for THREE knobs only, matching the solver playbook:
  drift_bp    center shift, basis points of |anchor| (or absolute bp for
              log_return targets), clamped to +-1.5 horizon sd downstream
  vol_scale   dispersion multiplier, [0.6, 2.0] requested / [0.7, 1.6] applied
  skew        asymmetric tail stretch in [-0.5, 0.5] (scenario mixture's
              cheap stand-in: widening one side only)
plus
  scenarios   optional labelled branches [{p, per-asset shift in sd}] that
              produce a real mixture over the joint draws
  anchors     per-asset level estimates ONLY for assets absent from the
              panels or with a withheld middle (transfer cards)

The endpoint contract: POST $MODEL_ENDPOINT/v1/chat/completions with
MODEL_NAME / MODEL_TOKEN through the audited proxy (urllib honours
HTTP_PROXY/HTTPS_PROXY). No third-party endpoints exist on the network;
a missing endpoint degrades to the statistical floor, labelled.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

MAX_DOC_CHARS = 3500
MAX_DOCS = 14
TIMEOUT = 75.0
OUT_TOKENS = 3500          # under the 4000 admitted cap


def endpoint() -> str | None:
    ep = os.environ.get("MODEL_ENDPOINT", "").strip()
    return ep.rstrip("/") + "/v1" if ep else None


def available() -> bool:
    return bool(endpoint() and os.environ.get("MODEL_NAME"))


def pick_docs(docs, asof: str) -> tuple[list, int]:
    """Admissible corpus docs (timestamp <= asof), newest first,
    deduplicated, capped for prompt budget."""
    seen, keep, skipped = set(), [], 0
    for d in sorted(docs, key=lambda x: (x.timestamp or "9999", x.doc_id),
                    reverse=True):
        if not d.timestamp or d.timestamp[:10] > asof or not d.text.strip():
            skipped += 1
            continue
        key = re.sub(r"\s+", " ", d.text.strip().lower())[:400]
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        keep.append(d)
    return keep[:MAX_DOCS], skipped


def build_prompt(bundle, docs, ctx: dict) -> str:
    """ctx: per-asset {anchor, sigma_h, steps} + horizon keys."""
    lines = [
        "You are adjusting a probabilistic forecast with dated documents.",
        f"As-of date: {bundle.asof}. Nothing after this date is known.",
        f"Target type: {bundle.target_type} "
        f"({bundle.value_unit or 'panel units'}), "
        f"frequency {bundle.target_frequency}.",
        f"Horizon keys: {bundle.horizons}.",
        "",
        "Per target asset: anchor (last observed level; 0 for log_return),",
        "forecast standard deviation at the LONGEST horizon, observation",
        "steps resolved, and whether its panel history is complete:",
    ]
    for a in bundle.target_assets:
        c = ctx.get(a, {})
        lines.append(
            f"  {a}: anchor={c.get('anchor', 'n/a'):.6g}, "
            f"sd_long_h={c.get('sigma_h', 'n/a'):.6g}, "
            f"steps={c.get('steps', '?')}, history={c.get('history', 'full')}")
    lines += ["", f"Documents ({len(docs)}), newest first:"]
    for d in docs:
        lines += [f"--- {d.doc_id} ({d.timestamp}, {d.doc_type}, {d.source}) ---",
                  d.text[:MAX_DOC_CHARS], ""]
    lines += [
        "Rules:",
        "- Use ONLY the supplied documents plus generic economic reasoning.",
        "  Never recall prices, levels, or event outcomes from memory.",
        "- 'drift_bp': expected directional shift over the LONGEST horizon,",
        "  in basis points of |anchor| (for log_return targets: absolute",
        "  return bp). 0 when the documents say nothing. Modest, not huge:",
        "  a big call is ~1 sd.",
        "- 'vol_scale': dispersion multiplier vs the statistical baseline.",
        "  >1 for crisis/binary-event risk inside the horizon, <1 only for",
        "  explicitly established calm. In [0.6, 2.0].",
        "- 'skew': in [-0.5, 0.5]; >0 widens the up side, <0 the down side.",
        "  Use it when the documents describe an asymmetric branch (peg,",
        "  vote, binary decision).",
        "- 'scenarios': optional, at most 3 named branches with probabilities",
        "  summing <= 1 and per-asset shifts in units of sd_long_h. Leave",
        "  empty when one regime is clearly dominant.",
        "- 'anchors': only for assets whose panel history ends long before",
        "  the as-of or is absent: your best point estimate of the CURRENT",
        "  level and its daily sd in percent, or null.",
        "- Every number finite; cite the doc_id behind each call in 'why'.",
        "",
        "Reply with JSON only:",
        '{"assets": {"<id>": {"drift_bp": <f>, "vol_scale": <f>,',
        '   "skew": <f>, "why": "doc_id: ..."}},',
        ' "scenarios": [{"p": <f>, "shifts_sd": {"<id>": <f>},',
        '   "vol": <f>, "label": "..."}],',
        ' "anchors": {"<id>": {"level_est": <f|null>, "daily_vol_pct": <f|null>}}}',
    ]
    return "\n".join(lines)


def chat(prompt: str, max_tokens: int = OUT_TOKENS) -> tuple[dict | None, str]:
    """One POST to the House route; no retry (a spent admission is spent).
    Returns (parsed_json, reason_if_failed)."""
    if not available():
        return None, "MODEL_ENDPOINT/MODEL_NAME unset"
    body = json.dumps({
        "model": os.environ["MODEL_NAME"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        endpoint() + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('MODEL_TOKEN', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:
        return None, f"house request failed ({type(e).__name__})"
    try:
        content = payload["choices"][0]["message"]["content"]
    except Exception:
        return None, "reply had no choices[0].message.content"
    if not isinstance(content, str):
        return None, "reply content was not a string"
    i, j = content.find("{"), content.rfind("}")
    if i < 0 or j <= i:
        return None, "reply contained no JSON object"
    try:
        return json.loads(content[i:j + 1]), ""
    except ValueError as e:
        return None, f"reply JSON did not parse: {e}"
