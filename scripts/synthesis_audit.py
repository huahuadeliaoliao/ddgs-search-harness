#!/usr/bin/env python3
"""Audit a draft synthesis against a DDGS full dossier."""

import argparse
import json
import re
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[a-z0-9]+")
CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
ENTITY_RE = re.compile(
    r"\b(?:[A-Za-z0-9][A-Za-z0-9._-]{1,}/)?"
    r"[A-Za-z][A-Za-z0-9._]*"
    r"(?:[-_.][A-Za-z0-9][A-Za-z0-9._]*)+\b"
)
STOPWORDS = {
    "about",
    "after",
    "also",
    "been",
    "could",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "most",
    "over",
    "than",
    "that",
    "their",
    "them",
    "they",
    "this",
    "those",
    "what",
    "when",
    "with",
    "your",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Audit a draft answer against a search_dossier full JSON file.",
    )
    parser.add_argument("--dossier", required=True, help="Path to full_dossier.json.")
    parser.add_argument("--answer", required=True, help="Path to a draft answer file.")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Print compact audit JSON for agent consumption.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=20,
        help="Maximum answer claims to audit.",
    )
    return parser.parse_args()


def tokenize(text: str) -> set[str]:
    """Tokenize for conservative lexical support checks."""
    return {
        token
        for token in WORD_RE.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def load_json(path_value: str) -> dict[str, Any]:
    """Load a JSON object."""
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--dossier must contain a JSON object.")
    return payload


def split_claims(answer: str, *, limit: int) -> list[str]:
    """Split a draft answer into claim-like sentences."""
    raw_parts = CLAIM_SPLIT_RE.split(answer.strip())
    claims = []
    for part in raw_parts:
        claim = " ".join(part.strip().split())
        if len(claim) < 24:
            continue
        tokens = tokenize(claim)
        if len(tokens) < 4:
            continue
        if re.search(r"\d", claim) or ENTITY_RE.search(claim) or len(tokens) >= 8:
            claims.append(claim)
        if len(claims) >= limit:
            break
    return claims


def evidence_chunks(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect traceable evidence chunks from dossier surfaces."""
    chunks = []
    for cluster in dossier.get("clusters", []):
        text = " ".join(
            (cluster.get("titles") or [])
            + (cluster.get("snippets") or [])
            + (cluster.get("topic_terms") or [])
        )
        if text.strip():
            chunks.append(
                {
                    "kind": "cluster",
                    "cluster_id": cluster.get("cluster_id"),
                    "url": cluster.get("canonical_url"),
                    "text": text,
                    "tokens": tokenize(text),
                }
            )

    for page in dossier.get("extracted_pages", []):
        text = " ".join(
            str(page.get(field) or "") for field in ("title", "url", "preview", "error")
        )
        if text.strip():
            chunks.append(
                {
                    "kind": "extract",
                    "cluster_id": page.get("cluster_id"),
                    "url": page.get("url"),
                    "text": text,
                    "tokens": tokenize(text),
                }
            )
    return chunks


def claim_support(claim: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the best lexical evidence support for a claim."""
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return {"score": 0.0, "trace": None}

    best_score = 0.0
    best_chunk = None
    for chunk in chunks:
        overlap = claim_tokens & chunk["tokens"]
        score = len(overlap) / len(claim_tokens)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    trace = None
    if best_chunk:
        trace = {
            "kind": best_chunk["kind"],
            "cluster_id": best_chunk["cluster_id"],
            "url": best_chunk["url"],
        }
    return {"score": round(best_score, 3), "trace": trace}


def candidate_mentioned(answer: str, candidate: dict[str, Any]) -> bool:
    """Check whether a candidate family appears in the answer."""
    lowered = answer.lower()
    names = [candidate.get("name") or ""]
    names.extend(candidate.get("surface_forms") or [])
    for name in names:
        clean = str(name).strip().lower()
        if clean and clean in lowered:
            return True
    return False


def build_synthesis_audit(
    dossier: dict[str, Any],
    answer: str,
    *,
    max_claims: int = 20,
) -> dict[str, Any]:
    """Build a deterministic post-search synthesis audit."""
    chunks = evidence_chunks(dossier)
    unsupported_claims = []
    weakly_supported_claims = []
    for claim in split_claims(answer, limit=max_claims):
        support = claim_support(claim, chunks)
        row = {"claim": claim, "support": support}
        if support["score"] < 0.18:
            unsupported_claims.append(row)
        elif support["score"] < 0.35:
            weakly_supported_claims.append(row)

    omitted = []
    for candidate in (
        dossier.get("candidate_audit", {}).get("unpromoted_candidates") or []
    ):
        if candidate_mentioned(answer, candidate):
            continue
        omitted.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "name": candidate.get("name"),
                "promotion_score": candidate.get("scores", {}).get("promotion_score"),
                "why_notable": candidate.get("why_notable"),
                "trace": candidate.get("trace"),
            }
        )

    required_actions = []
    for item in omitted[:5]:
        required_actions.append(
            f"include_or_exclude_unpromoted_candidate:{item['name']}"
        )
    if unsupported_claims:
        required_actions.append("support_or_remove_unsupported_claims")

    return {
        "status": {
            "requires_revision": bool(required_actions),
            "unsupported_claim_count": len(unsupported_claims),
            "weakly_supported_claim_count": len(weakly_supported_claims),
            "omitted_high_fit_candidate_count": len(omitted),
        },
        "unsupported_claims": unsupported_claims[:5],
        "weakly_supported_claims": weakly_supported_claims[:5],
        "omitted_high_fit_candidates": omitted[:8],
        "required_actions": required_actions,
        "trace": {
            "dossier_run_id": dossier.get("run_meta", {}).get("run_id"),
            "evidence_chunk_count": len(chunks),
        },
    }


def main() -> None:
    """Run the synthesis audit."""
    args = parse_args()
    dossier = load_json(args.dossier)
    answer = Path(args.answer).read_text(encoding="utf-8")
    audit = build_synthesis_audit(dossier, answer, max_claims=args.max_claims)
    if args.agent:
        print(json.dumps(audit, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
