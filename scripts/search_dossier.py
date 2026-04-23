#!/usr/bin/env python3
"""Build a DDGS-first layered search dossier for agents."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


WORD_RE = re.compile(r"[a-z0-9]+")
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
    "latest",
    "more",
    "most",
    "official",
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
LOW_SIGNAL_QUERY_TOKENS = {
    "api",
    "docs",
    "developer",
    "developers",
    "guide",
    "guides",
    "latest",
    "official",
    "openai",
    "reference",
}
TOPIC_NOISE_TOKENS = LOW_SIGNAL_QUERY_TOKENS | {
    "and",
    "article",
    "articles",
    "announcement",
    "announcements",
    "blog",
    "blogs",
    "client",
    "clients",
    "com",
    "documentation",
    "example",
    "examples",
    "for",
    "index",
    "javascript",
    "lang",
    "launch",
    "launched",
    "method",
    "methods",
    "news",
    "overview",
    "post",
    "posts",
    "python",
    "release",
    "releases",
    "resource",
    "resources",
    "sdk",
    "site",
    "state",
    "states",
    "the",
    "update",
    "updates",
    "using",
}
RRF_K = 60


@dataclass
class PassPlan:
    """A single DDGS pass to execute."""

    pass_id: str
    query: str
    query_label: str
    category: str
    backend: str
    timelimit: str | None
    max_results: int


def normalize_domain(value: str) -> str:
    """Normalize either a bare domain or a URL-like value."""
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    if "://" in cleaned:
        return domain_from_url(cleaned)
    return cleaned[4:] if cleaned.startswith("www.") else cleaned


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a structured DDGS dossier with ranked clusters, typed gaps, and extracted evidence.",
    )
    parser.add_argument("query", help="Primary search query.")
    parser.add_argument(
        "--freshness",
        choices=("auto", "evergreen", "current", "breaking"),
        default="auto",
        help="How sensitive the question is to recency.",
    )
    parser.add_argument(
        "--authority",
        choices=("any", "prefer_official", "official_only"),
        default="any",
        help="How strongly to prefer official sources.",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated DDGS categories. Defaults depend on freshness.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="DDGS backend selector passed to search methods.",
    )
    parser.add_argument(
        "--region",
        default="us-en",
        help="DDGS region, for example us-en or uk-en.",
    )
    parser.add_argument(
        "--safesearch",
        choices=("on", "moderate", "off"),
        default="moderate",
        help="DDGS safesearch mode.",
    )
    parser.add_argument(
        "--timelimit",
        default="",
        help="Optional DDGS timelimit override: d, w, m, or y.",
    )
    parser.add_argument(
        "--max-results-per-pass",
        type=int,
        default=6,
        help="Maximum DDGS results to request for each pass.",
    )
    parser.add_argument(
        "--extract-top-k",
        type=int,
        default=3,
        help="How many recommended reads to fetch with ddgs.extract().",
    )
    parser.add_argument(
        "--extract-format",
        choices=("text_markdown", "text_plain"),
        default="text_plain",
        help="ddgs.extract() output format.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Extra query variant to include. Repeatable.",
    )
    parser.add_argument(
        "--official-domain",
        action="append",
        default=[],
        help="Domain to treat as official. Repeatable.",
    )
    parser.add_argument(
        "--stdout-format",
        choices=("markdown", "json"),
        default="markdown",
        help="What to print to stdout.",
    )
    parser.add_argument(
        "--write-json",
        default="",
        help="Optional path to write the JSON dossier.",
    )
    parser.add_argument(
        "--write-markdown",
        default="",
        help="Optional path to write the Markdown dossier.",
    )
    return parser.parse_args()


def load_ddgs_class() -> Any:
    """Import DDGS with a few fallback search roots."""
    env_path = os.environ.get("DDGS_IMPORT_PATH", "").strip()
    if env_path:
        sys.path.insert(0, env_path)

    repo_root = Path(__file__).resolve().parents[1]
    sibling_ddgs = repo_root.parent / "ddgs"
    if (sibling_ddgs / "ddgs" / "__init__.py").exists():
        sys.path.insert(0, str(sibling_ddgs))

    try:
        from ddgs import DDGS
    except Exception as ex:  # noqa: BLE001
        msg = (
            "Unable to import ddgs. Install it with `pip install ddgs`, or set "
            "`DDGS_IMPORT_PATH` to a local ddgs checkout."
        )
        raise SystemExit(f"{msg}\nOriginal error: {ex!r}") from ex

    return DDGS


def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(tz=UTC).isoformat()


def tokenize(text: str) -> set[str]:
    """Extract a light token set."""
    return {
        token
        for token in WORD_RE.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def query_token_weights(query: str) -> dict[str, float]:
    """Downweight generic query tokens so topic-bearing tokens matter more."""
    weights = {}
    for token in tokenize(query):
        weights[token] = 0.4 if token in LOW_SIGNAL_QUERY_TOKENS else 1.0
    return weights


def weighted_coverage(query_weights: dict[str, float], haystack_tokens: set[str]) -> float:
    """Return weighted token coverage against one field."""
    if not query_weights:
        return 0.0
    total = sum(query_weights.values())
    if total == 0:
        return 0.0
    matched = sum(weight for token, weight in query_weights.items() if token in haystack_tokens)
    return matched / total


def stable_id(*parts: str) -> str:
    """Build a deterministic compact identifier."""
    joined = "||".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha1(joined).hexdigest()[:12]


def normalize_url(url: str) -> str:
    """Normalize a URL for clustering."""
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{netloc}{path}"


def domain_from_url(url: str) -> str:
    """Extract a normalized domain."""
    if not url:
        return ""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def root_domain(domain: str) -> str:
    """Return a simple root-domain approximation."""
    parts = [part for part in domain.split(".") if part]
    if len(parts) <= 2:
        return domain
    return ".".join(parts[-2:])


def normalize_title(text: str) -> str:
    """Normalize title text for fallback clustering."""
    return " ".join(WORD_RE.findall(text.lower()))


def url_path_tokens(url: str) -> list[str]:
    """Extract path tokens from a URL for topic analysis."""
    if not url:
        return []
    parsed = urlparse(url)
    return [
        token
        for token in WORD_RE.findall(parsed.path.replace("-", " ").replace("_", " "))
        if len(token) >= 3
    ]


def canonical_topic_token(token: str) -> str:
    """Collapse light plural and inflection variants for topic grouping."""
    value = token.lower()
    if len(value) > 4 and value.endswith("ies"):
        return f"{value[:-3]}y"
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def document_family_kind(url: str, title: str) -> str:
    """Classify a page into a coarse document-family kind."""
    lowered_url = url.lower()
    path = urlparse(url).path.lower() if url else ""
    lowered_title = title.lower()

    if "/api-reference/" in lowered_url or "/api/reference/" in lowered_url:
        return "api_reference"
    if "/changelog/" in path or "release notes" in lowered_title or "changelog" in lowered_title:
        return "changelog"
    if "/docs/guides/" in path or "/guides/" in path or " guide" in lowered_title:
        return "guide"
    if "/blog/" in path or "/index/" in path:
        return "blog"
    if "/news/" in path or lowered_title.startswith("news:") or " news " in f" {lowered_title} ":
        return "news"
    if "/docs/" in path:
        return "documentation"
    if not path or path == "/":
        return "homepage"

    first_segment = next((part for part in path.split("/") if part), "")
    return normalize_title(first_segment).replace(" ", "_") or "page"


def document_family_from_url(url: str, title: str) -> str:
    """Approximate a document family for near-duplicate grouping."""
    if not url:
        return normalize_title(title)[:80]
    parsed = urlparse(url)
    domain = domain_from_url(url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return domain
    family_parts = parts[:2]
    return f"{domain}/{'/'.join(family_parts)}"


def build_topic_terms(query: str, title: str, url: str, snippet: str, family_kind: str) -> list[str]:
    """Build a small set of salient topic terms for semantic grouping."""
    query_weights = query_token_weights(query)
    weighted_counts: Counter[str] = Counter()
    field_hits: dict[str, set[str]] = defaultdict(set)

    def add_tokens(tokens: list[str] | set[str], weight: float, field: str) -> None:
        for token in tokens:
            token = canonical_topic_token(token)
            if token in TOPIC_NOISE_TOKENS:
                continue
            weighted_counts[token] += weight
            field_hits[token].add(field)

    add_tokens(list(tokenize(title)), 3.0, "title")
    add_tokens(url_path_tokens(url), 2.2, "url")
    add_tokens(list(tokenize(snippet)), 1.0, "snippet")

    for token, weight in query_weights.items():
        token = canonical_topic_token(token)
        if token in TOPIC_NOISE_TOKENS:
            continue
        weighted_counts[token] += 1.5 * weight
        field_hits[token].add("query")

    family_tokens = {
        canonical_topic_token(token)
        for token in WORD_RE.findall(family_kind.replace("_", " "))
        if len(token) >= 3
    }
    for token in family_tokens:
        weighted_counts.pop(token, None)
        field_hits.pop(token, None)

    ranked = sorted(
        weighted_counts,
        key=lambda token: (
            weighted_counts[token] + 0.35 * len(field_hits[token]),
            len(field_hits[token]),
            token,
        ),
        reverse=True,
    )
    if ranked:
        return ranked[:5]

    fallback = [
        canonical_topic_token(token)
        for token in tokenize(query)
        if canonical_topic_token(token) not in TOPIC_NOISE_TOKENS
    ]
    if fallback:
        return fallback[:3]
    return [
        canonical_topic_token(token)
        for token in tokenize(title)
        if canonical_topic_token(token) not in TOPIC_NOISE_TOKENS
    ][:3]


def build_topic_signature(topic_terms: list[str], title: str) -> str:
    """Render topic terms into a human-readable signature."""
    if topic_terms:
        return " ".join(topic_terms[:4])
    return normalize_title(title)[:40]


def build_topic_cluster_key(topic_terms: list[str], family_kind: str, title: str) -> str:
    """Build a deterministic semantic topic-cluster key."""
    if topic_terms:
        return " ".join(topic_terms[:2])
    fallback = normalize_title(title)[:40]
    return fallback or family_kind


def topic_support_score(
    topic_cluster_size: int,
    topic_cluster_official_count: int,
    topic_cluster_domain_count: int,
) -> float:
    """Score how well a topic is supported across pages and domains."""
    value = (
        0.6 * min(topic_cluster_size, 4) / 4
        + 0.25 * min(topic_cluster_official_count, 3) / 3
        + 0.15 * min(topic_cluster_domain_count, 3) / 3
    )
    return round(min(1.0, value), 3)


def parse_categories(raw: str, freshness: str) -> list[str]:
    """Resolve the search categories."""
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    if freshness in {"current", "breaking"}:
        return ["text", "news"]
    return ["text"]


def derive_timelimit(freshness: str, explicit: str, category: str) -> str | None:
    """Choose a timelimit when appropriate."""
    if explicit:
        return explicit
    if freshness == "breaking":
        return "d" if category == "news" else "w"
    if freshness == "current":
        return "w"
    return None


def build_query_pack(
    query: str,
    freshness: str,
    authority: str,
    extra_variants: list[str],
    official_domains: list[str],
) -> list[tuple[str, str]]:
    """Build a ddgs-first query pack."""
    variants: list[tuple[str, str]] = [("canonical", query)]

    if freshness in {"current", "breaking"}:
        variants.append(("recency", f"{query} latest"))

    if authority in {"prefer_official", "official_only"}:
        variants.append(("official-hint", f"{query} official"))

    for domain in official_domains:
        variants.append((f"official-domain:{domain}", f"site:{domain} {query}"))

    for idx, variant in enumerate(extra_variants, start=1):
        variants.append((f"user-variant-{idx}", variant))

    deduped: list[tuple[str, str]] = []
    seen = set()
    for label, value in variants:
        normalized = value.strip()
        if normalized and normalized not in seen:
            deduped.append((label, normalized))
            seen.add(normalized)
    return deduped


def build_passes(args: argparse.Namespace) -> list[PassPlan]:
    """Build the execution plan."""
    categories = parse_categories(args.categories, args.freshness)
    query_pack = build_query_pack(
        args.query,
        args.freshness,
        args.authority,
        args.variant,
        args.official_domain,
    )

    passes: list[PassPlan] = []
    counter = 1
    for category in categories:
        for label, query in query_pack:
            passes.append(
                PassPlan(
                    pass_id=f"pass-{counter}",
                    query=query,
                    query_label=label,
                    category=category,
                    backend=args.backend,
                    timelimit=derive_timelimit(args.freshness, args.timelimit, category),
                    max_results=args.max_results_per_pass,
                )
            )
            counter += 1
    return passes


def execute_pass(ddgs: Any, plan: PassPlan, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run one DDGS pass."""
    method = getattr(ddgs, plan.category)
    kwargs: dict[str, Any] = {
        "max_results": plan.max_results,
        "backend": plan.backend,
    }
    if plan.category in {"text", "images", "news", "videos"}:
        kwargs["region"] = args.region
        kwargs["safesearch"] = args.safesearch
        if plan.timelimit:
            kwargs["timelimit"] = plan.timelimit
    elif plan.timelimit:
        kwargs["timelimit"] = plan.timelimit
    return list(method(plan.query, **kwargs))


def reciprocal_rank(rank_in_pass: int) -> float:
    """Return an RRF-style contribution for one hit."""
    return round(1.0 / (RRF_K + rank_in_pass), 6)


def score_authority(domain: str, official_domains: set[str]) -> float:
    """Score authority using explicit official-domain hints."""
    if not domain or not official_domains:
        return 0.0
    if domain in official_domains:
        return 1.0
    if any(domain.endswith(f".{candidate}") for candidate in official_domains):
        return 0.95
    root = root_domain(domain)
    if any(root == root_domain(candidate) for candidate in official_domains):
        return 0.5
    return 0.0


def score_answerability(query: str, title: str, url: str, snippet: str) -> float:
    """Score how directly a result appears to answer the query."""
    query_weights = query_token_weights(query)
    if not query_weights:
        return 0.0

    title_tokens = tokenize(title)
    url_like_tokens = tokenize(url.replace("://", " ").replace("/", " "))
    snippet_tokens = tokenize(snippet)
    title_hits = weighted_coverage(query_weights, title_tokens)
    url_hits = weighted_coverage(query_weights, url_like_tokens)
    snippet_hits = weighted_coverage(query_weights, snippet_tokens)
    primary_hits = max(title_hits, url_hits)
    phrase_bonus = 0.15 if query.lower() in f"{title} {url} {snippet}".lower() else 0.0
    title_bonus = 0.1 if set(query_weights).issubset(title_tokens) else 0.0
    score = 0.7 * primary_hits + 0.2 * min(title_hits + url_hits, 1.0) + 0.1 * snippet_hits
    if snippet_hits - primary_hits >= 0.35:
        score -= 0.15
    return round(max(0.0, min(1.0, score + phrase_bonus + title_bonus)), 3)


def score_topic_alignment(query: str, title: str, url: str, snippet: str) -> float:
    """Score whether the page itself, not just the snippet, is about the query topic."""
    query_weights = {
        token: weight
        for token, weight in query_token_weights(query).items()
        if token not in LOW_SIGNAL_QUERY_TOKENS
    }
    if not query_weights:
        query_weights = query_token_weights(query)
    if not query_weights:
        return 0.0

    title_hits = weighted_coverage(query_weights, tokenize(title))
    url_hits = weighted_coverage(query_weights, set(url_path_tokens(url)))
    snippet_hits = weighted_coverage(query_weights, tokenize(snippet))
    primary_hits = max(title_hits, url_hits)
    score = 0.6 * primary_hits + 0.25 * min(title_hits + url_hits, 1.0) + 0.15 * snippet_hits
    if primary_hits < 0.35 and snippet_hits >= 0.6:
        score -= 0.3
    if primary_hits == 0.0:
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 3)


def score_freshness(date_text: str) -> float:
    """Score recency when DDGS provides a date."""
    if not date_text:
        return 0.0
    try:
        dt = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
    except ValueError:
        return 0.1
    age = datetime.now(tz=UTC) - dt.astimezone(UTC)
    if age <= timedelta(days=1):
        return 1.0
    if age <= timedelta(days=7):
        return 0.7
    if age <= timedelta(days=30):
        return 0.4
    return 0.2


def score_extractability(url: str, category: str, domain: str) -> float:
    """Estimate whether ddgs.extract() is likely to yield useful content."""
    if not url.startswith(("http://", "https://")):
        return 0.0
    score = 0.85
    if category in {"images", "videos"}:
        score -= 0.35
    if domain in {"youtube.com", "youtu.be", "vimeo.com"}:
        score -= 0.35
    if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        score -= 0.45
    return round(max(0.0, min(1.0, score)), 3)


def normalize_result(
    raw: dict[str, Any],
    plan: PassPlan,
    official_domains: set[str],
    rank_in_pass: int,
) -> dict[str, Any]:
    """Normalize DDGS category-specific result shapes."""
    url = (
        raw.get("href")
        or raw.get("url")
        or raw.get("embed_url")
        or raw.get("image")
        or ""
    )
    url = normalize_url(str(url))
    domain = domain_from_url(url)
    title = str(raw.get("title") or raw.get("publisher") or url or "").strip()
    snippet = str(
        raw.get("body")
        or raw.get("description")
        or raw.get("content")
        or raw.get("info")
        or ""
    ).strip()
    source = str(raw.get("source") or raw.get("publisher") or raw.get("author") or domain).strip()
    date = str(raw.get("date") or raw.get("published") or "").strip()
    authority_score = score_authority(domain, official_domains)
    answerability_score = score_answerability(plan.query, title, url, snippet)
    topic_alignment_score = score_topic_alignment(plan.query, title, url, snippet)
    freshness_score = score_freshness(date)
    extractability_score = score_extractability(url, plan.category, domain)
    cluster_key = url or f"{domain}::{normalize_title(title)}"
    family_kind = document_family_kind(url, title)
    topic_terms = build_topic_terms(plan.query, title, url, snippet, family_kind)
    return {
        "result_id": f"{plan.pass_id}-r{stable_id(url, title, snippet)}",
        "pass_id": plan.pass_id,
        "query": plan.query,
        "query_label": plan.query_label,
        "category": plan.category,
        "backend": plan.backend,
        "rank_in_pass": rank_in_pass,
        "rrf_contribution": reciprocal_rank(rank_in_pass),
        "title": title,
        "url": url,
        "domain": domain,
        "root_domain": root_domain(domain) if domain else "",
        "snippet": snippet,
        "source": source,
        "date": date,
        "official": authority_score >= 0.95,
        "authority_score": authority_score,
        "answerability_score": answerability_score,
        "topic_alignment_score": topic_alignment_score,
        "freshness_score": freshness_score,
        "extractability_score": extractability_score,
        "cluster_key": cluster_key,
        "document_family": document_family_from_url(url, title),
        "document_family_kind": family_kind,
        "topic_terms": topic_terms,
        "topic_signature": build_topic_signature(topic_terms, title),
        "topic_cluster_key": build_topic_cluster_key(topic_terms, family_kind, title),
        "raw": raw,
    }


def score_corroboration(items: list[dict[str, Any]]) -> float:
    """Score how consistently a cluster reappears across passes."""
    pass_support = len({item["pass_id"] for item in items})
    label_support = len({item["query_label"] for item in items})
    category_support = len({item["category"] for item in items})
    value = (
        0.55 * min(pass_support, 4) / 4
        + 0.25 * min(label_support, 4) / 4
        + 0.20 * min(category_support, 2) / 2
    )
    return round(min(1.0, value), 3)


def cluster_sort_key(cluster: dict[str, Any], authority_mode: str, freshness_mode: str) -> tuple[float, ...]:
    """Sort clusters using explicit signal ordering instead of one opaque score."""
    signals = cluster["signal_scores"]
    authority_score = signals["authority"]
    answerability_score = signals["answerability"]
    freshness_score = signals["freshness"]
    corroboration_score = signals["corroboration"]
    extractability_score = signals["extractability"]
    fusion_score = cluster["rank_fusion_score"]
    topic_support = cluster.get("topic_support_score", 0.0)
    topic_alignment = cluster.get("topic_alignment_score", 0.0)

    if authority_mode in {"prefer_official", "official_only"} and freshness_mode in {"current", "breaking"}:
        return (
            authority_score,
            freshness_score,
            answerability_score,
            topic_alignment,
            topic_support,
            corroboration_score,
            fusion_score,
            extractability_score,
        )
    if authority_mode in {"prefer_official", "official_only"}:
        return (
            authority_score,
            answerability_score,
            topic_alignment,
            topic_support,
            corroboration_score,
            fusion_score,
            freshness_score,
            extractability_score,
        )
    if freshness_mode in {"current", "breaking"}:
        return (
            freshness_score,
            answerability_score,
            topic_alignment,
            topic_support,
            corroboration_score,
            fusion_score,
            authority_score,
            extractability_score,
        )
    return (
        answerability_score,
        topic_alignment,
        topic_support,
        corroboration_score,
        fusion_score,
        authority_score,
        freshness_score,
        extractability_score,
    )


def cluster_results(
    results: list[dict[str, Any]],
    authority_mode: str,
    freshness_mode: str,
) -> list[dict[str, Any]]:
    """Aggregate normalized results into candidate clusters."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[item["cluster_key"]].append(item)

    clusters = []
    for key, items in grouped.items():
        items.sort(
            key=lambda item: (
                item["authority_score"],
                item["answerability_score"],
                item["rrf_contribution"],
                item["freshness_score"],
            ),
            reverse=True,
        )
        best = items[0]
        supporting_results = sorted(
            [
                {
                    "pass_id": item["pass_id"],
                    "query_label": item["query_label"],
                    "category": item["category"],
                    "rank_in_pass": item["rank_in_pass"],
                    "rrf_contribution": item["rrf_contribution"],
                }
                for item in items
            ],
            key=lambda row: (int(row["pass_id"].split("-")[-1]), row["rank_in_pass"]),
        )

        rank_fusion_score = round(sum(item["rrf_contribution"] for item in items), 6)
        authority_score = max(item["authority_score"] for item in items)
        answerability_score = max(item["answerability_score"] for item in items)
        topic_alignment_score = max(item["topic_alignment_score"] for item in items)
        freshness_score = max(item["freshness_score"] for item in items)
        extractability_score = max(item["extractability_score"] for item in items)
        corroboration_score = score_corroboration(items)
        pass_ids = sorted(
            {item["pass_id"] for item in items},
            key=lambda value: int(value.split("-")[-1]),
        )
        query_labels = sorted({item["query_label"] for item in items})
        queries = sorted({item["query"] for item in items})
        categories = sorted({item["category"] for item in items})
        sources = sorted({item["source"] for item in items if item["source"]})
        topic_counter = Counter()
        for item in items:
            topic_counter.update(item["topic_terms"])
        topic_terms = [token for token, _ in topic_counter.most_common(5)]
        if not topic_terms:
            topic_terms = best["topic_terms"]

        titles = []
        seen_titles = set()
        for item in items:
            if item["title"] and item["title"] not in seen_titles:
                titles.append(item["title"])
                seen_titles.add(item["title"])

        snippets = []
        seen_snippets = set()
        for item in items:
            if item["snippet"] and item["snippet"] not in seen_snippets:
                snippets.append(item["snippet"][:260])
                seen_snippets.add(item["snippet"])
            if len(snippets) >= 3:
                break

        clusters.append(
            {
                "cluster_id": f"cluster-{stable_id(key)}",
                "cluster_key": key,
                "canonical_url": best["url"],
                "domain": best["domain"],
                "root_domain": best["root_domain"],
                "document_family": best["document_family"],
                "document_family_kind": best["document_family_kind"],
                "document_family_group": f"{best['root_domain']}::{best['document_family_kind']}",
                "document_family_size": 1,
                "document_family_group_size": 1,
                "topic_terms": topic_terms,
                "topic_signature": build_topic_signature(topic_terms, titles[0] if titles else best["canonical_url"]),
                "topic_cluster_key": build_topic_cluster_key(
                    topic_terms,
                    best["document_family_kind"],
                    titles[0] if titles else best["canonical_url"],
                ),
                "topic_cluster_size": 1,
                "topic_cluster_official_count": 1 if authority_score >= 0.95 else 0,
                "topic_cluster_domain_count": 1 if best["root_domain"] else 0,
                "topic_support_score": 0.0,
                "titles": titles,
                "categories": categories,
                "sources": sources,
                "pass_ids": pass_ids,
                "query_labels": query_labels,
                "queries": queries,
                "result_count": len(items),
                "official": authority_score >= 0.95,
                "rank_fusion_score": rank_fusion_score,
                "topic_alignment_score": round(topic_alignment_score, 3),
                "signal_scores": {
                    "authority": round(authority_score, 3),
                    "answerability": round(answerability_score, 3),
                    "freshness": round(freshness_score, 3),
                    "corroboration": corroboration_score,
                    "extractability": round(extractability_score, 3),
                },
                "supporting_results": supporting_results,
                "snippets": snippets,
            }
        )

    family_counts = Counter(cluster["document_family"] for cluster in clusters if cluster["document_family"])
    family_group_counts = Counter(
        cluster["document_family_group"] for cluster in clusters if cluster["document_family_group"]
    )
    topic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        topic_groups[cluster["topic_cluster_key"]].append(cluster)

    for cluster in clusters:
        cluster["document_family_size"] = family_counts.get(cluster["document_family"], 1)
        cluster["document_family_group_size"] = family_group_counts.get(cluster["document_family_group"], 1)

    for topic_key, topic_members in topic_groups.items():
        topic_cluster_size = len(topic_members)
        topic_cluster_official_count = len([item for item in topic_members if item["official"]])
        topic_cluster_domain_count = len(
            {item["root_domain"] for item in topic_members if item["root_domain"]}
        )
        support_score = topic_support_score(
            topic_cluster_size,
            topic_cluster_official_count,
            topic_cluster_domain_count,
        )
        for cluster in topic_members:
            cluster["topic_cluster_key"] = topic_key
            cluster["topic_cluster_size"] = topic_cluster_size
            cluster["topic_cluster_official_count"] = topic_cluster_official_count
            cluster["topic_cluster_domain_count"] = topic_cluster_domain_count
            cluster["topic_support_score"] = support_score

    clusters.sort(
        key=lambda cluster: cluster_sort_key(cluster, authority_mode, freshness_mode),
        reverse=True,
    )
    return clusters


def cluster_metric(cluster: dict[str, Any], name: str) -> float:
    """Read a ranking metric from a cluster."""
    if name == "rank_fusion_score":
        return cluster["rank_fusion_score"]
    if name == "topic_support_score":
        return cluster.get("topic_support_score", 0.0)
    if name == "topic_alignment_score":
        return cluster.get("topic_alignment_score", 0.0)
    if name == "update_intent_score":
        return update_intent_score(cluster)
    return cluster["signal_scores"][name]


def anchor_hint_score(cluster: dict[str, Any]) -> float:
    """Prefer reference-like anchors for official entry points."""
    url = (cluster.get("canonical_url") or "").lower()
    score = 0.0
    if "/api-reference/" in url or "/api/reference/" in url:
        score += 0.3
    if url.rstrip("/").endswith("/responses"):
        score += 0.25
    if "/reference/" in url:
        score += 0.15
    return round(min(1.0, score), 3)


def update_intent_score(cluster: dict[str, Any]) -> float:
    """Prefer page families that often carry concrete updates."""
    family_kind = cluster.get("document_family_kind") or ""
    if family_kind == "changelog":
        return 1.0
    if family_kind == "blog":
        return 0.85
    if family_kind == "news":
        return 0.75
    if family_kind == "guide":
        return 0.55
    if family_kind == "documentation":
        return 0.4
    if family_kind == "api_reference":
        return 0.3
    return 0.2


def fresh_update_candidate(cluster: dict[str, Any]) -> bool:
    """Reject recent results that are new but only weakly tied to the actual topic."""
    freshness_score = cluster["signal_scores"]["freshness"]
    answerability_score = cluster["signal_scores"]["answerability"]
    authority_score = cluster["signal_scores"]["authority"]
    topic_alignment = cluster.get("topic_alignment_score", 0.0)
    topic_support = cluster.get("topic_support_score", 0.0)

    return (
        freshness_score >= 0.4
        and answerability_score >= 0.5
        and topic_alignment >= 0.5
        and (
            topic_support >= 0.35
            or authority_score >= 0.85
            or topic_alignment >= 0.72
        )
    )


def choose_cluster(
    clusters: list[dict[str, Any]],
    metric_order: list[str],
    *,
    predicate: Any = None,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Pick the strongest cluster for a role."""
    exclude_ids = exclude_ids or set()
    candidates = [
        cluster
        for cluster in clusters
        if cluster["cluster_id"] not in exclude_ids and (predicate(cluster) if predicate else True)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda cluster: tuple(cluster_metric(cluster, metric) for metric in metric_order),
    )


def summarize_role(role: str, cluster: dict[str, Any], selection_reason: str) -> dict[str, Any]:
    """Serialize a selected role slot."""
    title = cluster["titles"][0] if cluster["titles"] else cluster["canonical_url"]
    return {
        "role": role,
        "cluster_id": cluster["cluster_id"],
        "title": title,
        "url": cluster["canonical_url"],
        "domain": cluster["domain"],
        "root_domain": cluster["root_domain"],
        "document_family": cluster["document_family"],
        "document_family_kind": cluster["document_family_kind"],
        "topic_signature": cluster["topic_signature"],
        "topic_cluster_key": cluster["topic_cluster_key"],
        "topic_support_score": cluster["topic_support_score"],
        "topic_alignment_score": cluster["topic_alignment_score"],
        "pass_ids": cluster["pass_ids"],
        "signal_scores": cluster["signal_scores"],
        "selection_reason": selection_reason,
    }


def build_recommended_reads(
    clusters: list[dict[str, Any]],
    authority_mode: str,
    freshness_mode: str,
) -> dict[str, dict[str, Any] | None]:
    """Assign explicit reading roles from the cluster list."""
    selected_ids: set[str] = set()
    selected_domains: set[str] = set()
    selected_topic_keys: set[str] = set()

    official_candidates = [
        cluster
        for cluster in clusters
        if cluster["signal_scores"]["authority"] >= 0.85
        and cluster["signal_scores"]["answerability"] >= 0.45
    ]
    official_anchor = None
    if official_candidates:
        official_anchor = max(
            official_candidates,
            key=lambda cluster: (
                anchor_hint_score(cluster),
                cluster["signal_scores"]["answerability"],
                cluster.get("topic_support_score", 0.0),
                cluster["rank_fusion_score"],
                cluster["signal_scores"]["corroboration"],
                cluster["signal_scores"]["freshness"],
            ),
        )
    if official_anchor is not None:
        selected_ids.add(official_anchor["cluster_id"])
        if official_anchor["root_domain"]:
            selected_domains.add(official_anchor["root_domain"])
        if official_anchor["topic_cluster_key"]:
            selected_topic_keys.add(official_anchor["topic_cluster_key"])

    best_direct_answer = choose_cluster(
        clusters,
        ["answerability", "authority", "topic_support_score", "corroboration", "rank_fusion_score", "freshness"]
        if authority_mode in {"prefer_official", "official_only"}
        else ["answerability", "topic_support_score", "corroboration", "authority", "rank_fusion_score", "freshness"],
        predicate=lambda cluster: cluster["signal_scores"]["answerability"] >= 0.55,
    )
    if (
        authority_mode in {"prefer_official", "official_only"}
        and official_anchor is not None
        and official_anchor["signal_scores"]["answerability"] >= 0.65
    ):
        best_direct_answer = official_anchor
    if best_direct_answer is not None:
        selected_ids.add(best_direct_answer["cluster_id"])
        if best_direct_answer["root_domain"]:
            selected_domains.add(best_direct_answer["root_domain"])
        if best_direct_answer["topic_cluster_key"]:
            selected_topic_keys.add(best_direct_answer["topic_cluster_key"])

    fresh_update = None
    if freshness_mode in {"current", "breaking"}:
        fresh_update = choose_cluster(
            clusters,
            [
                "freshness",
                "topic_alignment_score",
                "topic_support_score",
                "update_intent_score",
                "answerability",
                "authority",
                "corroboration",
                "rank_fusion_score",
            ],
            predicate=fresh_update_candidate,
        )
        if fresh_update is not None:
            selected_ids.add(fresh_update["cluster_id"])
            if fresh_update["root_domain"]:
                selected_domains.add(fresh_update["root_domain"])
            if fresh_update["topic_cluster_key"]:
                selected_topic_keys.add(fresh_update["topic_cluster_key"])

    background_context = choose_cluster(
        clusters,
        [
            "corroboration",
            "topic_alignment_score",
            "topic_support_score",
            "rank_fusion_score",
            "answerability",
            "authority",
            "freshness",
        ],
        exclude_ids=selected_ids,
        predicate=lambda cluster: (
            cluster["signal_scores"]["answerability"] >= 0.35
            and cluster.get("topic_alignment_score", 0.0) >= 0.25
        ),
    )
    if background_context is not None:
        selected_ids.add(background_context["cluster_id"])
        if background_context["root_domain"]:
            selected_domains.add(background_context["root_domain"])
        if background_context["topic_cluster_key"]:
            selected_topic_keys.add(background_context["topic_cluster_key"])

    alternate_view = choose_cluster(
        clusters,
        [
            "answerability",
            "topic_alignment_score",
            "topic_support_score",
            "rank_fusion_score",
            "corroboration",
            "freshness",
            "authority",
        ],
        exclude_ids=selected_ids,
        predicate=lambda cluster: (
            cluster["signal_scores"]["answerability"] >= 0.35
            and cluster.get("topic_alignment_score", 0.0) >= 0.25
            and cluster["root_domain"]
            and cluster["root_domain"] not in selected_domains
            and (
                not selected_topic_keys
                or cluster["topic_cluster_key"] not in selected_topic_keys
            )
            and (
                background_context is None
                or cluster["document_family"] != background_context["document_family"]
            )
        ),
    )

    return {
        "official_anchor": summarize_role(
            "official_anchor",
            official_anchor,
            "Best high-authority anchor for the query.",
        )
        if official_anchor
        else None,
        "best_direct_answer": summarize_role(
            "best_direct_answer",
            best_direct_answer,
            "Best direct-answer candidate after corroboration and fusion checks.",
        )
        if best_direct_answer
        else None,
        "fresh_update": summarize_role(
            "fresh_update",
            fresh_update,
            "Best recent candidate that still looks materially on-topic.",
        )
        if fresh_update
        else None,
        "background_context": summarize_role(
            "background_context",
            background_context,
            "Useful context page that broadens understanding without duplicating the anchor.",
        )
        if background_context
        else None,
        "alternate_view": summarize_role(
            "alternate_view",
            alternate_view,
            "Alternate perspective from a different domain family.",
        )
        if alternate_view
        else None,
    }


def extract_pages(
    ddgs: Any,
    recommended_reads: dict[str, dict[str, Any] | None],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Extract content for the strongest recommended reads."""
    extracted = []
    seen_urls = set()
    for role in (
        "official_anchor",
        "best_direct_answer",
        "fresh_update",
        "background_context",
        "alternate_view",
    ):
        candidate = recommended_reads.get(role)
        if not candidate:
            continue
        url = candidate["url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            payload = ddgs.extract(url, fmt=args.extract_format)
            content = payload["content"]
            if isinstance(content, bytes):
                preview = content[:800].decode("utf-8", errors="replace")
            else:
                preview = str(content)[:1600]
            extracted.append(
                {
                    "role": role,
                    "cluster_id": candidate["cluster_id"],
                    "url": url,
                    "title": candidate["title"],
                    "preview": preview.strip(),
                }
            )
        except Exception as ex:  # noqa: BLE001
            extracted.append(
                {
                    "role": role,
                    "cluster_id": candidate["cluster_id"],
                    "url": url,
                    "title": candidate["title"],
                    "error": f"{type(ex).__name__}: {ex}",
                }
            )
        if len(extracted) >= args.extract_top_k:
            break
    return extracted


def build_pass_yield(
    pass_records: list[dict[str, Any]],
    normalized_results: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach marginal-yield stats to each pass record."""
    pass_order = {
        record["pass_id"]: index
        for index, record in enumerate(pass_records)
    }
    pass_domains: dict[str, set[str]] = defaultdict(set)
    pass_rank_total: dict[str, int] = defaultdict(int)
    pass_rank_count: dict[str, int] = defaultdict(int)
    pass_touched: dict[str, set[str]] = defaultdict(set)
    pass_novel: dict[str, set[str]] = defaultdict(set)
    pass_novel_official: dict[str, set[str]] = defaultdict(set)
    pass_novel_fresh: dict[str, set[str]] = defaultdict(set)
    pass_novel_extractable: dict[str, set[str]] = defaultdict(set)

    for item in normalized_results:
        pass_id = item["pass_id"]
        if item["domain"]:
            pass_domains[pass_id].add(item["domain"])
        pass_rank_total[pass_id] += item["rank_in_pass"]
        pass_rank_count[pass_id] += 1

    for cluster in clusters:
        supporting = cluster["supporting_results"]
        if not supporting:
            continue
        introducer = min(
            supporting,
            key=lambda row: (pass_order.get(row["pass_id"], 9999), row["rank_in_pass"]),
        )
        introduced_by = introducer["pass_id"]
        for row in supporting:
            pass_touched[row["pass_id"]].add(cluster["cluster_id"])

        pass_novel[introduced_by].add(cluster["cluster_id"])
        if cluster["signal_scores"]["authority"] >= 0.85:
            pass_novel_official[introduced_by].add(cluster["cluster_id"])
        if cluster["signal_scores"]["freshness"] >= 0.4:
            pass_novel_fresh[introduced_by].add(cluster["cluster_id"])
        if cluster["signal_scores"]["extractability"] >= 0.65:
            pass_novel_extractable[introduced_by].add(cluster["cluster_id"])

    enriched = []
    for record in pass_records:
        pass_id = record["pass_id"]
        touched_count = len(pass_touched[pass_id])
        novel_count = len(pass_novel[pass_id])
        redundancy_ratio = 0.0 if touched_count == 0 else round(1 - (novel_count / touched_count), 3)
        mean_rank = 0.0
        if pass_rank_count[pass_id]:
            mean_rank = round(pass_rank_total[pass_id] / pass_rank_count[pass_id], 2)
        enriched.append(
            {
                **record,
                "yield_analysis": {
                    "touched_cluster_count": touched_count,
                    "novel_cluster_count": novel_count,
                    "novel_official_count": len(pass_novel_official[pass_id]),
                    "novel_fresh_count": len(pass_novel_fresh[pass_id]),
                    "novel_extractable_count": len(pass_novel_extractable[pass_id]),
                    "unique_domain_count": len(pass_domains[pass_id]),
                    "mean_rank_in_pass": mean_rank,
                    "redundancy_ratio": redundancy_ratio,
                },
            }
        )
    return enriched


def build_pass_yield_summary(pass_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Summarize which passes added the most or least new value."""
    def slim(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "pass_id": record["pass_id"],
                "category": record["category"],
                "query_label": record["query_label"],
                "query": record["query"],
                "novel_cluster_count": record["yield_analysis"]["novel_cluster_count"],
                "redundancy_ratio": record["yield_analysis"]["redundancy_ratio"],
            }
            for record in records
        ]

    novelty_sorted = sorted(
        pass_records,
        key=lambda record: (
            record["yield_analysis"]["novel_cluster_count"],
            -record["yield_analysis"]["redundancy_ratio"],
            record["yield_analysis"]["novel_official_count"],
        ),
        reverse=True,
    )
    redundancy_sorted = sorted(
        pass_records,
        key=lambda record: (
            record["yield_analysis"]["redundancy_ratio"],
            -record["yield_analysis"]["novel_cluster_count"],
        ),
        reverse=True,
    )

    notes = []
    best_novelty = novelty_sorted[:2]
    if best_novelty and best_novelty[0]["yield_analysis"]["novel_cluster_count"] > 0:
        labels = ", ".join(record["pass_id"] for record in best_novelty if record["yield_analysis"]["novel_cluster_count"] > 0)
        notes.append(f"Strong novelty came from {labels}.")
    high_redundancy = [
        record for record in redundancy_sorted
        if record["yield_analysis"]["redundancy_ratio"] >= 0.8 and record["yield_analysis"]["touched_cluster_count"] > 0
    ][:2]
    if high_redundancy:
        labels = ", ".join(record["pass_id"] for record in high_redundancy)
        notes.append(f"High redundancy suggests {labels} may be low-yield to repeat unchanged.")

    return {
        "best_novelty_passes": slim(best_novelty),
        "high_redundancy_passes": slim(high_redundancy),
        "notes": notes,
    }


def build_sufficiency_signals(
    clusters: list[dict[str, Any]],
    recommended_reads: dict[str, dict[str, Any] | None],
    extracted_pages: list[dict[str, Any]],
    passes: list[PassPlan],
    authority: str,
    freshness: str,
) -> dict[str, Any]:
    """Summarize dossier completeness without forcing a routing decision."""
    domain_diversity = len({cluster["root_domain"] for cluster in clusters if cluster["root_domain"]})
    topic_cluster_count = len({cluster["topic_cluster_key"] for cluster in clusters if cluster["topic_cluster_key"]})
    supported_topic_cluster_count = len(
        {
            cluster["topic_cluster_key"]
            for cluster in clusters
            if cluster["topic_cluster_key"] and cluster.get("topic_support_score", 0.0) >= 0.4
        }
    )
    official_cluster_count = len(
        [cluster for cluster in clusters if cluster["signal_scores"]["authority"] >= 0.85]
    )
    recent_cluster_count = len(
        [cluster for cluster in clusters if cluster["signal_scores"]["freshness"] >= 0.4]
    )
    direct_cluster_count = len(
        [cluster for cluster in clusters if cluster["signal_scores"]["answerability"] >= 0.5]
    )
    corroborated_cluster_count = len(
        [cluster for cluster in clusters if cluster["signal_scores"]["corroboration"] >= 0.35]
    )
    extracted_evidence_count = len([page for page in extracted_pages if "error" not in page])

    official_coverage = recommended_reads.get("official_anchor") is not None
    recency_coverage = (
        True if freshness not in {"current", "breaking"} else recommended_reads.get("fresh_update") is not None
    )
    direct_answer_coverage = recommended_reads.get("best_direct_answer") is not None
    requires_more_search = False
    if authority == "official_only" and not official_coverage:
        requires_more_search = True
    if freshness in {"current", "breaking"} and not recency_coverage:
        requires_more_search = True
    if not direct_answer_coverage or extracted_evidence_count == 0:
        requires_more_search = True

    return {
        "pass_count": len(passes),
        "cluster_count": len(clusters),
        "domain_diversity": domain_diversity,
        "topic_cluster_count": topic_cluster_count,
        "supported_topic_cluster_count": supported_topic_cluster_count,
        "official_cluster_count": official_cluster_count,
        "recent_cluster_count": recent_cluster_count,
        "direct_cluster_count": direct_cluster_count,
        "corroborated_cluster_count": corroborated_cluster_count,
        "official_coverage": official_coverage,
        "recency_coverage": recency_coverage,
        "direct_answer_coverage": direct_answer_coverage,
        "alternate_view_coverage": recommended_reads.get("alternate_view") is not None,
        "extracted_evidence_count": extracted_evidence_count,
        "requires_more_search": requires_more_search,
    }


def make_gap(
    gap_type: str,
    severity: str,
    reason: str,
    evidence: dict[str, Any],
    preferred_ddgs_move: str,
    native_search_candidate: bool,
) -> dict[str, Any]:
    """Build a typed gap record."""
    return {
        "gap_type": gap_type,
        "severity": severity,
        "reason": reason,
        "evidence": evidence,
        "preferred_ddgs_move": preferred_ddgs_move,
        "native_search_candidate": native_search_candidate,
    }


def build_open_gaps(
    signals: dict[str, Any],
    recommended_reads: dict[str, dict[str, Any] | None],
    extracted_pages: list[dict[str, Any]],
    authority: str,
    freshness: str,
) -> list[dict[str, Any]]:
    """Explain what is still missing in a typed, agent-friendly shape."""
    gaps = []

    if authority in {"prefer_official", "official_only"} and not recommended_reads.get("official_anchor"):
        gaps.append(
            make_gap(
                "authority_gap",
                "high" if authority == "official_only" else "medium",
                "The dossier has not surfaced a strong official anchor yet.",
                {
                    "official_cluster_count": signals["official_cluster_count"],
                    "authority_mode": authority,
                },
                "Add or tighten official-domain DDGS passes before changing tools.",
                True,
            )
        )

    if freshness in {"current", "breaking"} and not recommended_reads.get("fresh_update"):
        gaps.append(
            make_gap(
                "freshness_gap",
                "high",
                "The dossier lacks a recent on-topic update slot.",
                {
                    "recent_cluster_count": signals["recent_cluster_count"],
                    "supported_topic_cluster_count": signals["supported_topic_cluster_count"],
                    "freshness_mode": freshness,
                },
                "Rerun DDGS with a tighter timelimit and one recency-focused query variant.",
                True,
            )
        )

    direct_candidate = recommended_reads.get("best_direct_answer")
    if not direct_candidate:
        gaps.append(
            make_gap(
                "answer_gap",
                "high",
                "Results look related, but no cluster qualifies as a direct answer yet.",
                {
                    "direct_cluster_count": signals["direct_cluster_count"],
                },
                "Add one targeted query rewrite and extract more DDGS candidates before escalating.",
                True,
            )
        )

    extraction_errors = len([page for page in extracted_pages if "error" in page])
    if signals["extracted_evidence_count"] == 0:
        gaps.append(
            make_gap(
                "extraction_gap",
                "high",
                "The harness has not successfully pulled content-level evidence yet.",
                {
                    "extract_attempt_count": len(extracted_pages),
                    "extract_error_count": extraction_errors,
                },
                "Increase --extract-top-k or target more extractable DDGS candidates.",
                True,
            )
        )

    if signals["domain_diversity"] < 2:
        gaps.append(
            make_gap(
                "diversity_gap",
                "low",
                "Source diversity is still narrow, which weakens corroboration.",
                {
                    "domain_diversity": signals["domain_diversity"],
                    "alternate_view_present": signals["alternate_view_coverage"],
                },
                "Add one contrastive query variant to widen the DDGS result pool.",
                False,
            )
        )

    return gaps


def build_next_options(
    gaps: list[dict[str, Any]],
    authority: str,
) -> list[str]:
    """Suggest next retrieval moves without hardcoding routing."""
    options = []
    seen = set()

    for gap in gaps:
        move = gap["preferred_ddgs_move"]
        if move not in seen:
            options.append(move)
            seen.add(move)

    if any(gap["native_search_candidate"] for gap in gaps):
        message = "If the DDGS dossier still lacks the needed evidence, optionally supplement with native web_search."
        if message not in seen:
            options.append(message)
            seen.add(message)

    if authority == "official_only":
        message = "Discard non-official clusters when synthesizing the final answer."
        if message not in seen:
            options.append(message)

    return options


def render_markdown(dossier: dict[str, Any]) -> str:
    """Render a human-readable markdown dossier."""
    lines = [
        "# DDGS Search Dossier",
        "",
        "## Search Intent",
        f"- Query: {dossier['intent']['query']}",
        f"- Freshness: {dossier['intent']['freshness']}",
        f"- Authority: {dossier['intent']['authority']}",
        f"- Categories: {', '.join(dossier['intent']['categories'])}",
        f"- Region: {dossier['intent']['region']}",
        f"- Official domains: {', '.join(dossier['intent']['official_domains']) or 'none'}",
        "",
        "## Query Pack",
        f"- Canonical query: {dossier['query_pack']['canonical_query']}",
    ]
    for label, value in dossier["query_pack"]["variants"]:
        lines.append(f"- Variant [{label}]: {value}")

    lines.extend(["", "## DDGS Passes"])
    for item in dossier["passes"]:
        lines.extend(
            [
                f"### {item['pass_id']}",
                f"- Category: {item['category']}",
                f"- Query: {item['query']}",
                f"- Label: {item['query_label']}",
                f"- Backend: {item['backend']}",
                f"- Timelimit: {item['timelimit'] or 'none'}",
                f"- Result count: {item['result_count']}",
                f"- Yield: touched={item['yield_analysis']['touched_cluster_count']}, novel={item['yield_analysis']['novel_cluster_count']}, redundancy={item['yield_analysis']['redundancy_ratio']}",
                f"- Novel official/fresh/extractable: {item['yield_analysis']['novel_official_count']}/{item['yield_analysis']['novel_fresh_count']}/{item['yield_analysis']['novel_extractable_count']}",
            ]
        )
        if "error" in item:
            lines.append(f"- Error: {item['error']}")
        lines.append("")

    lines.append("## Pass Yield Summary")
    for heading, items in dossier["pass_yield_summary"].items():
        if heading == "notes":
            continue
        lines.append(f"### {heading}")
        if not items:
            lines.append("- none")
        else:
            for item in items:
                lines.append(
                    f"- {item['pass_id']} [{item['category']} / {item['query_label']}] novel={item['novel_cluster_count']} redundancy={item['redundancy_ratio']}"
                )
        lines.append("")
    if dossier["pass_yield_summary"]["notes"]:
        lines.append("### notes")
        for note in dossier["pass_yield_summary"]["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Candidate Clusters")
    for idx, cluster in enumerate(dossier["clusters"], start=1):
        title = cluster["titles"][0] if cluster["titles"] else cluster["canonical_url"]
        signals = cluster["signal_scores"]
        lines.extend(
            [
                f"### {idx}. {title}",
                f"- Domain: {cluster['domain'] or 'unknown'}",
                f"- URL: {cluster['canonical_url'] or 'missing'}",
                f"- Family: {cluster['document_family'] or 'unknown'}",
                f"- Family kind: {cluster['document_family_kind'] or 'unknown'}",
                f"- Family-group size: {cluster['document_family_group_size']}",
                f"- Topic signature: {cluster['topic_signature'] or 'unknown'}",
                f"- Topic cluster: {cluster['topic_cluster_key'] or 'unknown'}",
                f"- Topic support: {cluster['topic_support_score']}",
                f"- Topic alignment: {cluster['topic_alignment_score']}",
                f"- Official: {'yes' if cluster['official'] else 'no'}",
                f"- Categories: {', '.join(cluster['categories']) or 'none'}",
                f"- Query labels: {', '.join(cluster['query_labels']) or 'none'}",
                f"- Passes: {', '.join(cluster['pass_ids']) or 'none'}",
                f"- RRF fusion: {cluster['rank_fusion_score']}",
                f"- Signals: authority={signals['authority']}, answerability={signals['answerability']}, freshness={signals['freshness']}, corroboration={signals['corroboration']}, extractability={signals['extractability']}",
            ]
        )
        for support in cluster["supporting_results"][:4]:
            lines.append(
                f"- Support: {support['pass_id']} rank={support['rank_in_pass']} label={support['query_label']} category={support['category']}"
            )
        for snippet in cluster["snippets"]:
            lines.append(f"- Snippet: {snippet}")
        lines.append("")

    lines.append("## Recommended Reads")
    for role, candidate in dossier["recommended_reads"].items():
        lines.append(f"### {role}")
        if not candidate:
            lines.append("- none")
            lines.append("")
            continue
        signals = candidate["signal_scores"]
        lines.extend(
            [
                f"- Title: {candidate['title']}",
                f"- URL: {candidate['url']}",
                f"- Domain: {candidate['domain'] or 'unknown'}",
                f"- Family kind: {candidate['document_family_kind'] or 'unknown'}",
                f"- Topic cluster: {candidate['topic_cluster_key'] or 'unknown'}",
                f"- Reason: {candidate['selection_reason']}",
                f"- Topic support/alignment: {candidate['topic_support_score']}/{candidate['topic_alignment_score']}",
                f"- Signals: authority={signals['authority']}, answerability={signals['answerability']}, freshness={signals['freshness']}, corroboration={signals['corroboration']}, extractability={signals['extractability']}",
                "",
            ]
        )

    lines.append("## Extracted Evidence")
    if not dossier["extracted_pages"]:
        lines.append("- none")
        lines.append("")
    else:
        for page in dossier["extracted_pages"]:
            lines.append(f"### {page['role']}: {page['title']}")
            lines.append(f"- URL: {page['url']}")
            if "error" in page:
                lines.append(f"- Error: {page['error']}")
            else:
                lines.append(f"- Preview: {page['preview'].replace(chr(10), ' ')[:800]}")
            lines.append("")

    lines.extend(
        [
            "## Sufficiency Signals",
            f"- Pass count: {dossier['sufficiency_signals']['pass_count']}",
            f"- Cluster count: {dossier['sufficiency_signals']['cluster_count']}",
            f"- Domain diversity: {dossier['sufficiency_signals']['domain_diversity']}",
            f"- Topic cluster count: {dossier['sufficiency_signals']['topic_cluster_count']}",
            f"- Supported topic-cluster count: {dossier['sufficiency_signals']['supported_topic_cluster_count']}",
            f"- Official cluster count: {dossier['sufficiency_signals']['official_cluster_count']}",
            f"- Recent cluster count: {dossier['sufficiency_signals']['recent_cluster_count']}",
            f"- Direct cluster count: {dossier['sufficiency_signals']['direct_cluster_count']}",
            f"- Corroborated cluster count: {dossier['sufficiency_signals']['corroborated_cluster_count']}",
            f"- Official coverage: {dossier['sufficiency_signals']['official_coverage']}",
            f"- Recency coverage: {dossier['sufficiency_signals']['recency_coverage']}",
            f"- Direct-answer coverage: {dossier['sufficiency_signals']['direct_answer_coverage']}",
            f"- Alternate-view coverage: {dossier['sufficiency_signals']['alternate_view_coverage']}",
            f"- Extracted evidence count: {dossier['sufficiency_signals']['extracted_evidence_count']}",
            f"- Requires more search: {dossier['sufficiency_signals']['requires_more_search']}",
            "",
            "## Open Gaps",
        ]
    )
    if dossier["open_gaps"]:
        for gap in dossier["open_gaps"]:
            lines.extend(
                [
                    f"### {gap['gap_type']} [{gap['severity']}]",
                    f"- Reason: {gap['reason']}",
                    f"- Evidence: {json.dumps(gap['evidence'], ensure_ascii=False, sort_keys=True)}",
                    f"- Preferred DDGS move: {gap['preferred_ddgs_move']}",
                    f"- Native search candidate: {'yes' if gap['native_search_candidate'] else 'no'}",
                    "",
                ]
            )
    else:
        lines.extend(["- none", ""])

    lines.extend(["## Next Options"])
    if dossier["next_options"]:
        lines.extend(f"- {item}" for item in dossier["next_options"])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_optional(path_value: str, content: str) -> None:
    """Write content if a path was supplied."""
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Execute the dossier build."""
    args = parse_args()
    DDGS = load_ddgs_class()

    passes = build_passes(args)
    official_domains = {
        normalized for normalized in (normalize_domain(domain) for domain in args.official_domain) if normalized
    }

    pass_records = []
    normalized_results = []
    query_pack = build_query_pack(
        args.query,
        args.freshness,
        args.authority,
        args.variant,
        args.official_domain,
    )

    with DDGS() as ddgs:
        for plan in passes:
            try:
                raw_results = execute_pass(ddgs, plan, args)
            except Exception as ex:  # noqa: BLE001
                pass_records.append(
                    {
                        **asdict(plan),
                        "result_count": 0,
                        "error": f"{type(ex).__name__}: {ex}",
                    }
                )
                continue

            pass_records.append(
                {
                    **asdict(plan),
                    "result_count": len(raw_results),
                }
            )
            for rank_in_pass, raw in enumerate(raw_results, start=1):
                normalized_results.append(
                    normalize_result(raw, plan, official_domains, rank_in_pass)
                )

        clusters = cluster_results(normalized_results, args.authority, args.freshness)
        recommended_reads = build_recommended_reads(clusters, args.authority, args.freshness)
        extracted_pages = extract_pages(ddgs, recommended_reads, args)

    pass_records = build_pass_yield(pass_records, normalized_results, clusters)
    pass_yield_summary = build_pass_yield_summary(pass_records)
    signals = build_sufficiency_signals(
        clusters,
        recommended_reads,
        extracted_pages,
        passes,
        args.authority,
        args.freshness,
    )
    gaps = build_open_gaps(
        signals,
        recommended_reads,
        extracted_pages,
        args.authority,
        args.freshness,
    )
    next_options = build_next_options(gaps, args.authority)

    dossier = {
        "run_meta": {
            "created_at": now_iso(),
            "stdout_format": args.stdout_format,
            "harness_profile": "deterministic_multi_signal",
        },
        "intent": {
            "query": args.query,
            "freshness": args.freshness,
            "authority": args.authority,
            "categories": parse_categories(args.categories, args.freshness),
            "region": args.region,
            "safesearch": args.safesearch,
            "official_domains": args.official_domain,
        },
        "query_pack": {
            "canonical_query": args.query,
            "variants": query_pack,
        },
        "passes": pass_records,
        "pass_yield_summary": pass_yield_summary,
        "clusters": clusters,
        "recommended_reads": recommended_reads,
        "extracted_pages": extracted_pages,
        "sufficiency_signals": signals,
        "open_gaps": gaps,
        "next_options": next_options,
    }

    markdown = render_markdown(dossier)
    json_text = json.dumps(dossier, indent=2, ensure_ascii=False)

    write_optional(args.write_markdown, markdown)
    write_optional(args.write_json, json_text)

    if args.stdout_format == "json":
        print(json_text)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
