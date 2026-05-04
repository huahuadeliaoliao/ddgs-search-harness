#!/usr/bin/env python3
"""Build a DDGS-first layered search dossier for agents."""

import argparse
import hashlib
import json
import os
import re
import shlex
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
    "paper",
    "papers",
    "post",
    "posts",
    "python",
    "release",
    "releases",
    "resource",
    "resources",
    "result",
    "results",
    "review",
    "reviews",
    "sdk",
    "site",
    "state",
    "states",
    "table",
    "tables",
    "the",
    "update",
    "updates",
    "using",
}
CANDIDATE_NOISE_TOKENS = TOPIC_NOISE_TOKENS | {
    "abs",
    "arxiv",
    "blob",
    "card",
    "cards",
    "data",
    "download",
    "downloads",
    "file",
    "files",
    "fork",
    "forks",
    "github",
    "gitlab",
    "hf",
    "home",
    "html",
    "http",
    "https",
    "huggingface",
    "hub",
    "issues",
    "license",
    "licenses",
    "main",
    "master",
    "model",
    "models",
    "org",
    "page",
    "pages",
    "paperswithcode",
    "pull",
    "raw",
    "readme",
    "repo",
    "repos",
    "repository",
    "repositories",
    "src",
    "star",
    "stars",
    "task",
    "tasks",
    "tree",
    "www",
}
CANDIDATE_ALIAS_NOISE_TOKENS = CANDIDATE_NOISE_TOKENS | {
    "ai",
    "app",
    "lab",
    "labs",
    "open",
    "team",
}
CANDIDATE_ENTITY_RE = re.compile(
    r"\b(?:[A-Za-z0-9][A-Za-z0-9._-]{1,}/)?"
    r"[A-Za-z][A-Za-z0-9._]*"
    r"(?:[-_.][A-Za-z0-9][A-Za-z0-9._]*)+\b"
)
CAMEL_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}(?:[A-Z][A-Za-z0-9]{2,})+\b")
RRF_K = 60


@dataclass
class QueryFacet:
    """A typed query input with provenance."""

    facet: str
    query: str
    label: str
    source: str = "agent"
    memory_seeded: bool = False
    exploration_anchor: bool = False
    operator_intensity: int = 0
    notes: str = ""


@dataclass
class PassPlan:
    """A single DDGS pass to execute."""

    pass_id: str
    query: str
    query_label: str
    query_facet: str
    memory_seeded: bool
    exploration_anchor: bool
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
        "--agent",
        action="store_true",
        help="Print compact agent packet to stdout and write full trace artifacts.",
    )
    parser.add_argument(
        "--query-plan",
        default="",
        help="Optional JSON file containing concept map, query facets, stress profiles, and hypotheses.",
    )
    parser.add_argument(
        "--query-facet",
        action="append",
        default=[],
        help="Typed query facet in FACET:QUERY form. Repeatable.",
    )
    parser.add_argument(
        "--hypothesis-entity",
        action="append",
        default=[],
        help="Remembered or tentative entity to quarantine as a hypothesis. Repeatable.",
    )
    parser.add_argument(
        "--user-entity",
        action="append",
        default=[],
        help="Entity explicitly supplied by the user. Repeatable.",
    )
    parser.add_argument(
        "--stress-profile",
        action="append",
        choices=(
            "single_hard_fact",
            "current_research",
            "web_traversal",
            "exhaustive_list",
            "deep_research_report",
            "multilingual_web",
            "multimodal_browsing",
            "enterprise_or_private_corpus",
        ),
        default=[],
        help="Task stress profile. Repeatable.",
    )
    parser.add_argument(
        "--budget",
        choices=("tiny", "normal", "deep"),
        default="normal",
        help="Agent packet context budget.",
    )
    parser.add_argument(
        "--top-sources",
        type=int,
        default=0,
        help="Override number of sources in the agent packet.",
    )
    parser.add_argument(
        "--top-harvest",
        type=int,
        default=0,
        help="Override number of harvest candidates in the agent packet.",
    )
    parser.add_argument(
        "--top-unpromoted",
        type=int,
        default=0,
        help="Override number of unpromoted candidates in the agent packet.",
    )
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=0,
        help="Override extracted excerpt length in the agent packet.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Directory for trace artifacts. Defaults to output/runs/<run_id> in agent mode.",
    )
    parser.add_argument(
        "--stdout-format",
        choices=("markdown", "json", "agent_packet"),
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


def weighted_coverage(
    query_weights: dict[str, float], haystack_tokens: set[str]
) -> float:
    """Return weighted token coverage against one field."""
    if not query_weights:
        return 0.0
    total = sum(query_weights.values())
    if total == 0:
        return 0.0
    matched = sum(
        weight for token, weight in query_weights.items() if token in haystack_tokens
    )
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
    if (
        "/changelog/" in path
        or "release notes" in lowered_title
        or "changelog" in lowered_title
    ):
        return "changelog"
    if "/docs/guides/" in path or "/guides/" in path or " guide" in lowered_title:
        return "guide"
    if "/blog/" in path or "/index/" in path:
        return "blog"
    if (
        "/news/" in path
        or lowered_title.startswith("news:")
        or " news " in f" {lowered_title} "
    ):
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


def build_topic_terms(
    query: str, title: str, url: str, snippet: str, family_kind: str
) -> list[str]:
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


def build_topic_cluster_key(
    topic_terms: list[str], family_kind: str, title: str
) -> str:
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


ALLOWED_QUERY_FACETS = {
    "task_anchor",
    "domain_broadening",
    "source_surface",
    "recency_probe",
    "contrastive",
    "hypothesis_entity",
    "user_entity",
    "harvest_refinement",
    "legacy_variant",
}


def operator_intensity(query: str) -> int:
    """Count query operators that narrow exploration."""
    lowered = query.lower()
    patterns = (
        "site:",
        "filetype:",
        "before:",
        "after:",
        '"',
        "-",
        "intitle:",
        "inurl:",
    )
    return sum(1 for pattern in patterns if pattern in lowered)


def load_query_plan(path_value: str) -> dict[str, Any]:
    """Load an optional query-plan JSON file."""
    if not path_value:
        return {}
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        raise SystemExit(f"Unable to read --query-plan {path}: {ex}") from ex
    if not isinstance(payload, dict):
        raise SystemExit("--query-plan must contain a JSON object.")
    return payload


def unique_list(values: list[Any]) -> list[Any]:
    """Keep first occurrence ordering for simple JSON values."""
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result


def parse_query_facet(raw: str) -> dict[str, Any]:
    """Parse FACET:QUERY syntax into a query facet object."""
    if ":" not in raw:
        raise SystemExit("--query-facet must use FACET:QUERY syntax.")
    facet, query = raw.split(":", 1)
    facet = facet.strip()
    query = query.strip()
    if facet not in ALLOWED_QUERY_FACETS:
        allowed = ", ".join(sorted(ALLOWED_QUERY_FACETS))
        raise SystemExit(f"Unknown query facet {facet!r}. Expected one of: {allowed}.")
    if not query:
        raise SystemExit("--query-facet query must not be empty.")
    return {"facet": facet, "query": query, "source": "cli_query_facet"}


def build_query_strategy(args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI and optional plan inputs into a serializable strategy object."""
    plan = load_query_plan(args.query_plan)
    concept_map = dict(plan.get("concept_map") or {})
    plan_hypotheses = list(
        plan.get("hypothesis_entities") or concept_map.get("hypothesis_entities") or []
    )
    plan_user_entities = list(
        plan.get("user_entities") or concept_map.get("user_entities") or []
    )
    hypothesis_entities = unique_list(plan_hypotheses + list(args.hypothesis_entity))
    user_entities = unique_list(plan_user_entities + list(args.user_entity))
    concept_map["hypothesis_entities"] = hypothesis_entities
    concept_map["user_entities"] = user_entities

    query_facets = []
    for item in plan.get("query_facets") or []:
        if not isinstance(item, dict):
            raise SystemExit("Each query-plan query_facets item must be an object.")
        query_facets.append({**item, "source": item.get("source") or "query_plan"})
    query_facets.extend(parse_query_facet(item) for item in args.query_facet)

    stress_profiles = unique_list(
        list(plan.get("stress_profiles") or []) + list(args.stress_profile)
    )
    return {
        "stress_profiles": stress_profiles,
        "concept_map": concept_map,
        "query_facets": query_facets,
        "hypothesis_entities": hypothesis_entities,
        "user_entities": user_entities,
        "strategy_warnings": [],
    }


def facet_from_dict(item: dict[str, Any], idx: int) -> QueryFacet:
    """Convert a query-facet object into a QueryFacet."""
    facet = str(item.get("facet") or "legacy_variant").strip()
    query = str(item.get("query") or "").strip()
    if facet not in ALLOWED_QUERY_FACETS:
        allowed = ", ".join(sorted(ALLOWED_QUERY_FACETS))
        raise SystemExit(f"Unknown query facet {facet!r}. Expected one of: {allowed}.")
    if not query:
        raise SystemExit("Query facet objects must include a non-empty query.")

    memory_seeded = bool(item.get("memory_seeded", facet == "hypothesis_entity"))
    exploration_anchor = bool(
        item.get(
            "exploration_anchor",
            facet in {"task_anchor", "domain_broadening"} and not memory_seeded,
        )
    )
    label = str(item.get("label") or f"{facet}-{idx}").strip()
    return QueryFacet(
        facet=facet,
        query=query,
        label=label,
        source=str(item.get("source") or "agent"),
        memory_seeded=memory_seeded,
        exploration_anchor=exploration_anchor,
        operator_intensity=int(
            item.get("operator_intensity", operator_intensity(query))
        ),
        notes=str(item.get("notes") or ""),
    )


def exploration_anchors(facets: list[QueryFacet]) -> list[QueryFacet]:
    """Return neutral facets that are allowed to seed derived queries."""
    return [
        facet
        for facet in facets
        if facet.exploration_anchor and not facet.memory_seeded
    ]


def has_exploration_anchor(facets: list[QueryFacet]) -> bool:
    """Whether a query pack has a non-memory exploration anchor."""
    return bool(exploration_anchors(facets))


def dedupe_facets(facets: list[QueryFacet]) -> list[QueryFacet]:
    """Deduplicate query facets by query text and facet kind."""
    deduped = []
    seen = set()
    for facet in facets:
        normalized_query = " ".join(facet.query.split())
        if not normalized_query:
            continue
        key = (facet.facet, normalized_query.lower())
        if key in seen:
            continue
        deduped.append(
            QueryFacet(
                facet=facet.facet,
                query=normalized_query,
                label=facet.label,
                source=facet.source,
                memory_seeded=facet.memory_seeded,
                exploration_anchor=facet.exploration_anchor,
                operator_intensity=facet.operator_intensity,
                notes=facet.notes,
            )
        )
        seen.add(key)
    return deduped


def build_query_pack(
    args: argparse.Namespace,
    query_strategy: dict[str, Any],
) -> list[QueryFacet]:
    """Build a provenance-aware DDGS query pack."""
    facets = [
        facet_from_dict(item, idx)
        for idx, item in enumerate(query_strategy.get("query_facets") or [], start=1)
    ]

    if not has_exploration_anchor(facets):
        facets.append(
            QueryFacet(
                facet="task_anchor",
                query=args.query,
                label="task-anchor",
                source="positional_query",
                memory_seeded=False,
                exploration_anchor=True,
                operator_intensity=operator_intensity(args.query),
            )
        )

    anchors = exploration_anchors(facets)

    if args.freshness in {"current", "breaking"}:
        for anchor in anchors:
            facets.append(
                QueryFacet(
                    facet="recency_probe",
                    query=f"{anchor.query} latest",
                    label=f"recency:{anchor.label}",
                    source="derived",
                    memory_seeded=False,
                    exploration_anchor=False,
                    operator_intensity=operator_intensity(anchor.query),
                )
            )

    if args.authority in {"prefer_official", "official_only"}:
        for anchor in anchors:
            facets.append(
                QueryFacet(
                    facet="source_surface",
                    query=f"{anchor.query} official",
                    label=f"official-hint:{anchor.label}",
                    source="derived",
                    memory_seeded=False,
                    exploration_anchor=False,
                    operator_intensity=operator_intensity(anchor.query),
                )
            )

    for domain in args.official_domain:
        normalized_domain = normalize_domain(domain)
        if not normalized_domain:
            continue
        for anchor in anchors:
            query = f"site:{normalized_domain} {anchor.query}"
            facets.append(
                QueryFacet(
                    facet="source_surface",
                    query=query,
                    label=f"source-surface:{normalized_domain}",
                    source="official_domain",
                    memory_seeded=False,
                    exploration_anchor=False,
                    operator_intensity=operator_intensity(query),
                )
            )

    for idx, variant in enumerate(args.variant, start=1):
        facets.append(
            QueryFacet(
                facet="legacy_variant",
                query=variant,
                label=f"legacy-variant-{idx}",
                source="legacy_variant",
                memory_seeded=False,
                exploration_anchor=False,
                operator_intensity=operator_intensity(variant),
            )
        )

    for entity in query_strategy.get("hypothesis_entities") or []:
        entity_text = str(entity).strip()
        if not entity_text:
            continue
        facets.append(
            QueryFacet(
                facet="hypothesis_entity",
                query=f"{entity_text} {args.query}",
                label=f"hypothesis:{entity_text}",
                source="agent_hypothesis",
                memory_seeded=True,
                exploration_anchor=False,
                operator_intensity=operator_intensity(entity_text),
            )
        )

    for entity in query_strategy.get("user_entities") or []:
        entity_text = str(entity).strip()
        if not entity_text:
            continue
        facets.append(
            QueryFacet(
                facet="user_entity",
                query=f"{entity_text} {args.query}",
                label=f"user-entity:{entity_text}",
                source="user_entity",
                memory_seeded=False,
                exploration_anchor=False,
                operator_intensity=operator_intensity(entity_text),
            )
        )

    return dedupe_facets(facets)


def has_language_or_region_variation(
    facets: list[QueryFacet], args: argparse.Namespace
) -> bool:
    """Detect whether multilingual/regional scope is represented."""
    if args.region and args.region != "us-en":
        return True
    joined = " ".join(facet.query for facet in facets).lower()
    markers = (
        " chinese ",
        " zh ",
        "中文",
        "japanese",
        "korean",
        "spanish",
        "regional",
        "local",
    )
    return any(marker in f" {joined} " for marker in markers)


def build_query_strategy_warnings(
    facets: list[QueryFacet],
    stress_profiles: list[str],
    args: argparse.Namespace,
    hypothesis_entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expose query-grounding risks before synthesis."""
    warnings = []
    hypothesis_entities = [
        str(item).strip() for item in (hypothesis_entities or []) if str(item).strip()
    ]
    if not has_exploration_anchor(facets):
        warnings.append(
            {
                "type": "query_grounding_gap",
                "severity": "high",
                "reason": "No neutral task or domain exploration anchor is present.",
                "next_move": "Add a task_anchor or domain_broadening facet before entity-specific search.",
            }
        )

    contaminated_anchors = []
    for anchor in exploration_anchors(facets):
        anchor_query = anchor.query.lower()
        if any(entity.lower() in anchor_query for entity in hypothesis_entities):
            contaminated_anchors.append(anchor.query)
    if contaminated_anchors:
        warnings.append(
            {
                "type": "query_grounding_gap",
                "severity": "high",
                "reason": "An exploration anchor contains a quarantined hypothesis entity.",
                "next_move": "Replace the canonical query with neutral task or domain terms and keep the entity in hypothesis_entity.",
            }
        )
        warnings.append(
            {
                "type": "entity_lock_in_gap",
                "severity": "high",
                "reason": "A remembered hypothesis entity is shaping the exploration anchor.",
                "next_move": "Search category-level terms first, then run entity-specific checks as hypothesis facets.",
            }
        )

    memory_count = len([facet for facet in facets if facet.memory_seeded])
    if facets and memory_count / len(facets) >= 0.5:
        warnings.append(
            {
                "type": "entity_lock_in_gap",
                "severity": "high",
                "reason": "Too many query facets are memory-seeded hypothesis entities.",
                "next_move": "Add neutral task, domain, source-surface, or contrastive facets.",
            }
        )

    anchors = exploration_anchors(facets)
    if anchors and all(facet.operator_intensity >= 2 for facet in anchors):
        warnings.append(
            {
                "type": "operator_overfit_gap",
                "severity": "medium",
                "reason": "Exploration anchors are already narrowed by multiple operators.",
                "next_move": "Add at least one neutral broad query without site, quote, filetype, minus, or date operators.",
            }
        )

    if "multilingual_web" in stress_profiles and not has_language_or_region_variation(
        facets, args
    ):
        warnings.append(
            {
                "type": "language_scope_gap",
                "severity": "medium",
                "reason": "The task may require multilingual or regional sources, but the query plan is single-language.",
                "next_move": "Add local-language or region-specific facets.",
            }
        )
    return warnings


def build_passes(
    args: argparse.Namespace, query_pack: list[QueryFacet]
) -> list[PassPlan]:
    """Build the execution plan."""
    categories = parse_categories(args.categories, args.freshness)

    passes: list[PassPlan] = []
    counter = 1
    for category in categories:
        for facet in query_pack:
            passes.append(
                PassPlan(
                    pass_id=f"pass-{counter}",
                    query=facet.query,
                    query_label=facet.label,
                    query_facet=facet.facet,
                    memory_seeded=facet.memory_seeded,
                    exploration_anchor=facet.exploration_anchor,
                    category=category,
                    backend=args.backend,
                    timelimit=derive_timelimit(
                        args.freshness, args.timelimit, category
                    ),
                    max_results=args.max_results_per_pass,
                )
            )
            counter += 1
    return passes


def execute_pass(
    ddgs: Any, plan: PassPlan, args: argparse.Namespace
) -> list[dict[str, Any]]:
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
    score = (
        0.7 * primary_hits + 0.2 * min(title_hits + url_hits, 1.0) + 0.1 * snippet_hits
    )
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
    score = (
        0.6 * primary_hits
        + 0.25 * min(title_hits + url_hits, 1.0)
        + 0.15 * snippet_hits
    )
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
    source = str(
        raw.get("source") or raw.get("publisher") or raw.get("author") or domain
    ).strip()
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
        "query_facet": plan.query_facet,
        "memory_seeded_query": plan.memory_seeded,
        "exploration_anchor_query": plan.exploration_anchor,
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


def cluster_sort_key(
    cluster: dict[str, Any], authority_mode: str, freshness_mode: str
) -> tuple[float, ...]:
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

    if authority_mode in {"prefer_official", "official_only"} and freshness_mode in {
        "current",
        "breaking",
    }:
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
                    "query_facet": item["query_facet"],
                    "memory_seeded": item["memory_seeded_query"],
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
        query_facets = sorted({item["query_facet"] for item in items})
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
                "topic_signature": build_topic_signature(
                    topic_terms, titles[0] if titles else best["canonical_url"]
                ),
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
                "query_facets": query_facets,
                "memory_seeded_support_count": len(
                    [item for item in items if item["memory_seeded_query"]]
                ),
                "exploration_anchor_support_count": len(
                    [item for item in items if item["exploration_anchor_query"]]
                ),
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

    family_counts = Counter(
        cluster["document_family"] for cluster in clusters if cluster["document_family"]
    )
    family_group_counts = Counter(
        cluster["document_family_group"]
        for cluster in clusters
        if cluster["document_family_group"]
    )
    topic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        topic_groups[cluster["topic_cluster_key"]].append(cluster)

    for cluster in clusters:
        cluster["document_family_size"] = family_counts.get(
            cluster["document_family"], 1
        )
        cluster["document_family_group_size"] = family_group_counts.get(
            cluster["document_family_group"], 1
        )

    for topic_key, topic_members in topic_groups.items():
        topic_cluster_size = len(topic_members)
        topic_cluster_official_count = len(
            [item for item in topic_members if item["official"]]
        )
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
            topic_support >= 0.35 or authority_score >= 0.85 or topic_alignment >= 0.72
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
        if cluster["cluster_id"] not in exclude_ids
        and (predicate(cluster) if predicate else True)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda cluster: tuple(
            cluster_metric(cluster, metric) for metric in metric_order
        ),
    )


def summarize_role(
    role: str, cluster: dict[str, Any], selection_reason: str
) -> dict[str, Any]:
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
        "query_facets": cluster.get("query_facets", []),
        "memory_seeded_support_count": cluster.get("memory_seeded_support_count", 0),
        "exploration_anchor_support_count": cluster.get(
            "exploration_anchor_support_count", 0
        ),
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
        [
            "answerability",
            "authority",
            "topic_support_score",
            "corroboration",
            "rank_fusion_score",
            "freshness",
        ]
        if authority_mode in {"prefer_official", "official_only"}
        else [
            "answerability",
            "topic_support_score",
            "corroboration",
            "authority",
            "rank_fusion_score",
            "freshness",
        ],
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
    pass_order = {record["pass_id"]: index for index, record in enumerate(pass_records)}
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
        redundancy_ratio = (
            0.0 if touched_count == 0 else round(1 - (novel_count / touched_count), 3)
        )
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


def build_pass_yield_summary(
    pass_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Summarize which passes added the most or least new value."""

    def slim(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "pass_id": record["pass_id"],
                "category": record["category"],
                "query_label": record["query_label"],
                "query_facet": record.get("query_facet", ""),
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
        labels = ", ".join(
            record["pass_id"]
            for record in best_novelty
            if record["yield_analysis"]["novel_cluster_count"] > 0
        )
        notes.append(f"Strong novelty came from {labels}.")
    high_redundancy = [
        record
        for record in redundancy_sorted
        if record["yield_analysis"]["redundancy_ratio"] >= 0.8
        and record["yield_analysis"]["touched_cluster_count"] > 0
    ][:2]
    if high_redundancy:
        labels = ", ".join(record["pass_id"] for record in high_redundancy)
        notes.append(
            f"High redundancy suggests {labels} may be low-yield to repeat unchanged."
        )

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
    domain_diversity = len(
        {cluster["root_domain"] for cluster in clusters if cluster["root_domain"]}
    )
    topic_cluster_count = len(
        {
            cluster["topic_cluster_key"]
            for cluster in clusters
            if cluster["topic_cluster_key"]
        }
    )
    supported_topic_cluster_count = len(
        {
            cluster["topic_cluster_key"]
            for cluster in clusters
            if cluster["topic_cluster_key"]
            and cluster.get("topic_support_score", 0.0) >= 0.4
        }
    )
    official_cluster_count = len(
        [
            cluster
            for cluster in clusters
            if cluster["signal_scores"]["authority"] >= 0.85
        ]
    )
    recent_cluster_count = len(
        [
            cluster
            for cluster in clusters
            if cluster["signal_scores"]["freshness"] >= 0.4
        ]
    )
    direct_cluster_count = len(
        [
            cluster
            for cluster in clusters
            if cluster["signal_scores"]["answerability"] >= 0.5
        ]
    )
    corroborated_cluster_count = len(
        [
            cluster
            for cluster in clusters
            if cluster["signal_scores"]["corroboration"] >= 0.35
        ]
    )
    extracted_evidence_count = len(
        [page for page in extracted_pages if "error" not in page]
    )

    official_coverage = recommended_reads.get("official_anchor") is not None
    recency_coverage = (
        True
        if freshness not in {"current", "breaking"}
        else recommended_reads.get("fresh_update") is not None
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


def normalize_harvest_term(value: str) -> str:
    """Normalize a candidate refinement term."""
    value = re.sub(r"\s+", " ", value.strip(" -_|:/.,()[]{}"))
    return value[:80]


def useful_harvest_term(value: str, query_tokens: set[str] | None = None) -> bool:
    """Filter out noisy harvested terms."""
    if len(value) < 3 or len(value) > 80:
        return False
    lowered = value.lower()
    if lowered in STOPWORDS or lowered in TOPIC_NOISE_TOKENS:
        return False
    if value.islower() and len(value) <= 3:
        return False
    term_tokens = tokenize(value)
    if query_tokens and term_tokens and term_tokens.issubset(query_tokens):
        return False
    if lowered.startswith(("http", "www.")):
        return False
    if len(value.split()) > 6:
        return False
    return bool(re.search(r"[a-zA-Z0-9]", value))


def extract_title_phrases(titles: list[str]) -> list[str]:
    """Extract deterministic title phrases for guarded refinement."""
    phrases = []
    for title in titles:
        for match in re.findall(
            r"\b[A-Z][A-Za-z0-9.]*[A-Za-z0-9](?:[- ][A-Z0-9][A-Za-z0-9.]*){0,3}\b",
            title,
        ):
            phrases.append(match)
        for match in re.findall(r"\b[A-Za-z]+(?:[- ][A-Za-z0-9]+){1,3}\b", title):
            if any(char.isdigit() for char in match):
                phrases.append(match)
    return phrases


def extract_url_entity_hints(url: str) -> list[str]:
    """Use URL path fragments as weak project/entity hints."""
    hints = []
    for token in url_path_tokens(url):
        if len(token) >= 3 and token not in TOPIC_NOISE_TOKENS:
            hints.append(token)
    return hints[:5]


def infer_harvest_kind(term: str, cluster: dict[str, Any]) -> str:
    """Assign a lightweight kind to a harvested term."""
    lowered = term.lower()
    if "benchmark" in lowered or "leaderboard" in lowered:
        return "benchmark_or_leaderboard"
    if cluster.get("root_domain") == "github.com":
        return "repo_or_project"
    if cluster.get("document_family_kind") in {"paper", "arxiv"}:
        return "paper_or_model_candidate"
    if any(char.isdigit() for char in term) or any(char.isupper() for char in term[1:]):
        return "entity_candidate"
    return "topic_term"


def source_surface_for_domain(root: str, official: bool = False) -> str:
    """Classify broad source surfaces without domain-specific candidate boosts."""
    if root in {"github.com", "gitlab.com"}:
        return "repo"
    if root == "huggingface.co":
        return "model_or_dataset"
    if root == "arxiv.org":
        return "paper"
    if root == "paperswithcode.com":
        return "leaderboard_or_index"
    if official:
        return "official"
    return "web"


def split_candidate_tokens(value: str) -> list[str]:
    """Split entity-like names into comparable lowercase tokens."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = value.replace("/", " ").replace("_", " ").replace("-", " ")
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", value)
        if len(token) >= 2 and token.lower() not in CANDIDATE_ALIAS_NOISE_TOKENS
    ]


def candidate_core_tokens(value: str) -> set[str]:
    """Return comparable tokens for alias linking."""
    return {canonical_topic_token(token) for token in split_candidate_tokens(value)}


def normalize_candidate_mention(value: str) -> str:
    """Clean a candidate mention while preserving useful entity casing."""
    value = value.strip(" \t\r\n.,:;()[]{}<>\"'")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"/+", "/", value)
    return value[:120]


def candidate_specificity_score(value: str) -> float:
    """Score whether a mention looks like a specific entity rather than a topic."""
    normalized = normalize_candidate_mention(value)
    if not normalized:
        return 0.0
    tokens = candidate_core_tokens(normalized)
    if not tokens:
        return 0.0

    score = 0.0
    if "/" in normalized:
        score += 0.25
    if re.search(r"\d", normalized):
        score += 0.2
    if any(separator in normalized for separator in ("-", "_", ".")):
        score += 0.15
    if re.search(r"[a-z][A-Z]", normalized):
        score += 0.2
    if re.search(r"\b[A-Z0-9]{2,}\b", normalized):
        score += 0.1
    if len(tokens) >= 2:
        score += 0.1
    if len(tokens) == 1 and next(iter(tokens)) in CANDIDATE_NOISE_TOKENS:
        score -= 0.3
    return round(max(0.0, min(1.0, score)), 3)


def useful_candidate_mention(value: str, task_tokens: set[str] | None = None) -> bool:
    """Keep entity-like mentions; reject generic topic fragments."""
    normalized = normalize_candidate_mention(value)
    if len(normalized) < 4 or len(normalized) > 120:
        return False
    lowered = normalized.lower()
    if lowered in CANDIDATE_NOISE_TOKENS:
        return False
    tokens = candidate_core_tokens(normalized)
    if not tokens:
        return False
    if tokens.issubset(CANDIDATE_NOISE_TOKENS):
        return False
    if (
        task_tokens
        and tokens.issubset(task_tokens)
        and candidate_specificity_score(normalized) < 0.45
    ):
        return False
    return candidate_specificity_score(normalized) >= 0.25


def extract_entity_phrases(text: str) -> list[str]:
    """Extract project/model/paper-like phrases from a text field."""
    if not text:
        return []
    phrases = []
    phrases.extend(match.group(0) for match in CANDIDATE_ENTITY_RE.finditer(text))
    phrases.extend(match.group(0) for match in CAMEL_ENTITY_RE.finditer(text))
    return unique_list([normalize_candidate_mention(item) for item in phrases])


def url_candidate_forms(url: str) -> list[str]:
    """Extract entity-like forms from source URLs."""
    if not url:
        return []
    parsed = urlparse(url)
    root = root_domain(domain_from_url(url))
    parts = [
        part
        for part in parsed.path.split("/")
        if part and part not in {"blob", "tree", "raw", "main", "master"}
    ]
    forms = []
    if root in {"github.com", "gitlab.com", "huggingface.co"} and len(parts) >= 2:
        forms.append(f"{parts[0]}/{parts[1]}")
    forms.extend(parts[:3])
    return unique_list([normalize_candidate_mention(item) for item in forms])


def cluster_candidate_mentions(
    cluster: dict[str, Any],
    task_tokens: set[str],
) -> list[dict[str, Any]]:
    """Extract candidate mentions from one cluster."""
    root = cluster.get("root_domain") or root_domain(cluster.get("domain") or "")
    surface = source_surface_for_domain(root, bool(cluster.get("official")))
    texts = []
    texts.extend(cluster.get("titles") or [])
    texts.extend(cluster.get("snippets") or [])
    texts.append(cluster.get("canonical_url") or "")

    raw_mentions = []
    for text in texts:
        raw_mentions.extend(extract_entity_phrases(text))
    raw_mentions.extend(url_candidate_forms(cluster.get("canonical_url") or ""))
    raw_mentions.extend(cluster.get("topic_terms") or [])

    mentions = []
    for raw in unique_list(raw_mentions):
        mention = normalize_candidate_mention(str(raw))
        if not useful_candidate_mention(mention, task_tokens):
            continue
        mentions.append(
            {
                "name": mention,
                "cluster_id": cluster["cluster_id"],
                "title": (cluster.get("titles") or [""])[0],
                "url": cluster.get("canonical_url") or "",
                "root_domain": root,
                "source_surface": surface,
                "official": bool(cluster.get("official")),
                "topic_alignment_score": cluster.get("topic_alignment_score", 0.0),
                "topic_support_score": cluster.get("topic_support_score", 0.0),
                "signal_scores": cluster.get("signal_scores") or {},
                "query_facets": cluster.get("query_facets") or [],
                "text": " ".join(texts)[:1600],
            }
        )
    return mentions


def mentions_alias_related(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Decide whether two mentions likely describe the same candidate family."""
    left_name = left["name"].lower()
    right_name = right["name"].lower()
    if left_name == right_name:
        return True
    if min(len(left_name), len(right_name)) >= 5 and (
        left_name in right_name or right_name in left_name
    ):
        return True

    left_tokens = candidate_core_tokens(left["name"])
    right_tokens = candidate_core_tokens(right["name"])
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens & right_tokens
    if len(shared) < 2:
        return False
    smaller = min(len(left_tokens), len(right_tokens))
    larger = max(len(left_tokens), len(right_tokens))
    if len(shared) == smaller and smaller >= 2:
        return True
    return len(shared) / larger >= 0.6


def merge_candidate_mentions(
    mentions: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group candidate mentions into deterministic alias families."""
    families: list[list[dict[str, Any]]] = []
    for mention in mentions:
        matched_indexes = [
            idx
            for idx, family in enumerate(families)
            if any(mentions_alias_related(mention, item) for item in family)
        ]
        if not matched_indexes:
            families.append([mention])
            continue

        first = matched_indexes[0]
        families[first].append(mention)
        for idx in reversed(matched_indexes[1:]):
            families[first].extend(families.pop(idx))
    return families


def candidate_display_name(family: list[dict[str, Any]]) -> str:
    """Pick the most specific readable name for a candidate family."""
    names = unique_list([item["name"] for item in family])
    return max(
        names,
        key=lambda name: (
            candidate_specificity_score(name),
            len(candidate_core_tokens(name)),
            len(name),
        ),
    )


def query_context_tokens(query: str, query_strategy: dict[str, Any]) -> set[str]:
    """Collect task terms from the user query, facets, and concept map."""
    values = [query]
    for facet in query_strategy.get("query_facets") or []:
        values.append(str(facet.get("query") or ""))
    concept_map = query_strategy.get("concept_map") or {}
    for item in concept_map.values():
        if isinstance(item, list):
            values.extend(str(value) for value in item)
        elif isinstance(item, str):
            values.append(item)
    return {
        canonical_topic_token(token)
        for token in tokenize(" ".join(values))
        if canonical_topic_token(token) not in TOPIC_NOISE_TOKENS
    }


def score_candidate_promotion(
    family: list[dict[str, Any]],
    *,
    task_tokens: set[str],
    promoted_cluster_ids: set[str],
    freshness: str,
) -> dict[str, Any]:
    """Score whether an unpromoted candidate deserves agent attention."""
    name = candidate_display_name(family)
    family_text = " ".join(
        [name]
        + [item.get("text") or "" for item in family]
        + [item.get("title") or "" for item in family]
    )
    candidate_tokens = {
        canonical_topic_token(token)
        for token in tokenize(family_text)
        if canonical_topic_token(token) not in TOPIC_NOISE_TOKENS
    } | candidate_core_tokens(name)

    task_fit = (
        len(task_tokens & candidate_tokens) / max(1, min(len(task_tokens), 8))
        if task_tokens
        else 0.0
    )
    source_surfaces = {item["source_surface"] for item in family}
    official_support = any(item.get("official") for item in family)
    source_surface_fit = min(1.0, len(source_surfaces) / 3)
    authority_fit = 1.0 if official_support else 0.0
    if not official_support and source_surfaces & {"repo", "model_or_dataset", "paper"}:
        authority_fit = 0.65
    specificity_fit = max(candidate_specificity_score(item["name"]) for item in family)
    freshness_fit = 0.0
    if freshness in {"current", "breaking"}:
        freshness_fit = max(
            item.get("signal_scores", {}).get("freshness", 0.0) for item in family
        )
    promoted = bool({item["cluster_id"] for item in family} & promoted_cluster_ids)
    novelty_fit = 0.0 if promoted else 0.35

    generic_noise_penalty = 0.0
    name_tokens = candidate_core_tokens(name)
    if task_tokens and name_tokens and name_tokens.issubset(task_tokens):
        generic_noise_penalty += 0.45
    if specificity_fit < 0.35:
        generic_noise_penalty += 0.25
    if len(name_tokens) <= 1:
        generic_noise_penalty += 0.15

    promotion_score = (
        0.32 * min(1.0, task_fit)
        + 0.18 * source_surface_fit
        + 0.16 * authority_fit
        + 0.18 * specificity_fit
        + 0.08 * freshness_fit
        + 0.08 * novelty_fit
        - 0.22 * min(1.0, generic_noise_penalty)
    )
    return {
        "task_fit": round(min(1.0, task_fit), 3),
        "source_surface_fit": round(source_surface_fit, 3),
        "authority_fit": round(authority_fit, 3),
        "specificity_fit": round(specificity_fit, 3),
        "freshness_fit": round(freshness_fit, 3),
        "novelty_fit": round(novelty_fit, 3),
        "generic_noise_penalty": round(min(1.0, generic_noise_penalty), 3),
        "promotion_score": round(max(0.0, min(1.0, promotion_score)), 3),
        "promoted": promoted,
    }


def render_candidate_family(
    family: list[dict[str, Any]],
    score: dict[str, Any],
) -> dict[str, Any]:
    """Serialize one candidate family for the dossier."""
    cluster_ids = sorted({item["cluster_id"] for item in family})
    source_surfaces = sorted({item["source_surface"] for item in family})
    root_domains = sorted(
        {item["root_domain"] for item in family if item["root_domain"]}
    )
    surface_forms = sorted(unique_list([item["name"] for item in family]))
    name = candidate_display_name(family)
    return {
        "candidate_id": f"candidate-{stable_id(name, *cluster_ids)}",
        "name": name,
        "kind": infer_harvest_kind(
            name, {"root_domain": "", "document_family_kind": ""}
        ),
        "surface_forms": surface_forms[:10],
        "source_surfaces": source_surfaces,
        "root_domains": root_domains,
        "scores": score,
        "official_support": any(item.get("official") for item in family),
        "trace": {
            "cluster_ids": cluster_ids[:8],
            "query_facets": sorted(
                {facet for item in family for facet in item.get("query_facets", [])}
            ),
        },
        "why_notable": candidate_notability_reason(source_surfaces, score),
    }


def candidate_notability_reason(
    source_surfaces: list[str], score: dict[str, Any]
) -> str:
    """Create a short agent-facing explanation for candidate attention."""
    reasons = []
    if score["task_fit"] >= 0.35:
        reasons.append("task-fit")
    if len(source_surfaces) >= 2:
        reasons.append("multi-surface")
    if score["authority_fit"] >= 0.65:
        reasons.append("authority-surface")
    if score["specificity_fit"] >= 0.55:
        reasons.append("specific-entity")
    if not reasons:
        reasons.append("retrieved-candidate")
    return "High-fit retrieved candidate flagged by " + ", ".join(reasons) + "."


def build_candidate_audit(
    clusters: list[dict[str, Any]],
    recommended_reads: dict[str, dict[str, Any] | None],
    query_strategy: dict[str, Any],
    *,
    query: str,
    freshness: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Find high-fit retrieved candidates that were not promoted."""
    task_tokens = query_context_tokens(query, query_strategy)
    promoted_cluster_ids = {
        candidate["cluster_id"]
        for candidate in recommended_reads.values()
        if candidate and candidate.get("cluster_id")
    }
    mentions = [
        mention
        for cluster in clusters
        for mention in cluster_candidate_mentions(cluster, task_tokens)
    ]
    families = []
    for family in merge_candidate_mentions(mentions):
        score = score_candidate_promotion(
            family,
            task_tokens=task_tokens,
            promoted_cluster_ids=promoted_cluster_ids,
            freshness=freshness,
        )
        rendered = render_candidate_family(family, score)
        families.append(rendered)

    families.sort(
        key=lambda item: (
            item["scores"]["promotion_score"],
            item["scores"]["task_fit"],
            item["scores"]["specificity_fit"],
            len(item["source_surfaces"]),
        ),
        reverse=True,
    )
    unpromoted = [
        item
        for item in families
        if not item["scores"]["promoted"]
        and (
            item["scores"]["promotion_score"] >= 0.48
            or (
                item["official_support"]
                and item["scores"]["task_fit"] >= 0.2
                and item["scores"]["specificity_fit"] >= 0.4
            )
        )
    ][:8]
    merge_warnings = [
        {
            "type": "cross_surface_candidate_not_promoted",
            "candidate_id": item["candidate_id"],
            "name": item["name"],
            "source_surfaces": item["source_surfaces"],
            "trace": item["trace"],
        }
        for item in unpromoted
        if len(item["source_surfaces"]) >= 2
    ]
    return {
        "candidate_families": families[:limit],
        "unpromoted_candidates": unpromoted,
        "merge_warnings": merge_warnings,
        "promotion_thresholds": {
            "default_unpromoted": 0.48,
            "official_specific_min_task_fit": 0.2,
            "official_specific_min_specificity": 0.4,
        },
    }


def build_harvest_candidates(
    clusters: list[dict[str, Any]],
    *,
    query: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Build guarded vocabulary candidates for optional refinement."""
    candidates: dict[str, dict[str, Any]] = {}
    query_tokens = tokenize(query)
    for cluster in clusters:
        signals = cluster["signal_scores"]
        topic_alignment = cluster.get("topic_alignment_score", 0.0)
        topic_support = cluster.get("topic_support_score", 0.0)
        if topic_alignment < 0.45 and signals["answerability"] < 0.45:
            continue

        terms = []
        terms.extend(cluster.get("topic_terms", [])[:5])
        terms.extend(extract_title_phrases(cluster.get("titles", [])))
        terms.extend(extract_url_entity_hints(cluster.get("canonical_url", "")))

        for term in terms:
            normalized = normalize_harvest_term(term)
            if not useful_harvest_term(normalized, query_tokens):
                continue
            record = candidates.setdefault(
                normalized.lower(),
                {
                    "term": normalized,
                    "kind": infer_harvest_kind(normalized, cluster),
                    "cluster_ids": set(),
                    "domains": set(),
                    "official_support": False,
                    "best_signal": 0.0,
                },
            )
            record["cluster_ids"].add(cluster["cluster_id"])
            if cluster["root_domain"]:
                record["domains"].add(cluster["root_domain"])
            record["official_support"] = (
                record["official_support"] or cluster["official"]
            )
            record["best_signal"] = max(
                record["best_signal"],
                signals["answerability"],
                topic_alignment,
                topic_support,
            )

    rendered = []
    for item in candidates.values():
        risk_flags = []
        if len(item["cluster_ids"]) == 1 and not item["official_support"]:
            risk_flags.append("single_cluster_support")
        if len(item["domains"]) < 2 and not item["official_support"]:
            risk_flags.append("low_domain_diversity")

        rendered.append(
            {
                "term": item["term"],
                "kind": item["kind"],
                "support": {
                    "cluster_count": len(item["cluster_ids"]),
                    "domain_count": len(item["domains"]),
                    "official_support": item["official_support"],
                    "best_signal": round(item["best_signal"], 3),
                },
                "risk_flags": risk_flags,
                "trace": {
                    "cluster_ids": sorted(item["cluster_ids"])[:5],
                },
            }
        )

    rendered.sort(
        key=lambda row: (
            row["support"]["official_support"],
            row["support"]["cluster_count"],
            row["support"]["domain_count"],
            row["support"]["best_signal"],
        ),
        reverse=True,
    )
    return rendered[:limit]


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
    *,
    strategy_warnings: list[dict[str, Any]] | None = None,
    harvest_candidates: list[dict[str, Any]] | None = None,
    candidate_audit: dict[str, Any] | None = None,
    stress_profiles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Explain what is still missing in a typed, agent-friendly shape."""
    gaps = []
    strategy_warnings = strategy_warnings or []
    harvest_candidates = harvest_candidates or []
    candidate_audit = candidate_audit or {}
    stress_profiles = stress_profiles or []

    for warning in strategy_warnings:
        gaps.append(
            make_gap(
                warning["type"],
                warning["severity"],
                warning["reason"],
                {"source": "query_strategy"},
                warning["next_move"],
                False,
            )
        )

    if authority in {"prefer_official", "official_only"} and not recommended_reads.get(
        "official_anchor"
    ):
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

    if freshness in {"current", "breaking"} and not recommended_reads.get(
        "fresh_update"
    ):
        gaps.append(
            make_gap(
                "freshness_gap",
                "high",
                "The dossier lacks a recent on-topic update slot.",
                {
                    "recent_cluster_count": signals["recent_cluster_count"],
                    "supported_topic_cluster_count": signals[
                        "supported_topic_cluster_count"
                    ],
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

    drift_risks = [
        item
        for item in harvest_candidates
        if item.get("risk_flags")
        and not item.get("support", {}).get("official_support")
    ]
    if drift_risks:
        gaps.append(
            make_gap(
                "harvest_drift_risk",
                "low",
                "Some refinement terms have weak support and may cause topic drift.",
                {
                    "risky_harvest_count": len(drift_risks),
                    "sample_terms": [item["term"] for item in drift_risks[:3]],
                },
                "Use only guarded harvest terms with topic-aligned, repeated, or official support.",
                False,
            )
        )

    unpromoted_candidates = candidate_audit.get("unpromoted_candidates") or []
    if unpromoted_candidates:
        gaps.append(
            make_gap(
                "unpromoted_candidate_gap",
                "medium",
                "High-fit candidates were retrieved but not promoted into the compact source set.",
                {
                    "candidate_count": len(unpromoted_candidates),
                    "sample_candidates": [
                        item["name"] for item in unpromoted_candidates[:3]
                    ],
                },
                "Inspect unpromoted_candidates before final synthesis; include, exclude with reason, or run targeted refinement.",
                False,
            )
        )

    official_unpromoted = [
        item for item in unpromoted_candidates if item.get("official_support")
    ]
    if official_unpromoted:
        gaps.append(
            make_gap(
                "official_hit_not_promoted_gap",
                "medium",
                "Official or authority-surface candidate evidence was retrieved but not promoted.",
                {
                    "candidate_count": len(official_unpromoted),
                    "sample_candidates": [
                        item["name"] for item in official_unpromoted[:3]
                    ],
                },
                "Review the candidate trace before relying on the current recommended reads.",
                False,
            )
        )

    merge_warnings = candidate_audit.get("merge_warnings") or []
    if merge_warnings:
        gaps.append(
            make_gap(
                "entity_merge_gap",
                "medium",
                "Candidate evidence appears across multiple source surfaces but was not promoted.",
                {
                    "candidate_count": len(merge_warnings),
                    "sample_candidates": [item["name"] for item in merge_warnings[:3]],
                },
                "Inspect merged candidate families before final synthesis.",
                False,
            )
        )

    if "web_traversal" in stress_profiles and signals["extracted_evidence_count"] == 0:
        gaps.append(
            make_gap(
                "traversal_gap",
                "medium",
                "The task may require navigating multi-level websites; DDGS snippets and single-page extraction may be insufficient.",
                {"extracted_evidence_count": signals["extracted_evidence_count"]},
                "Inspect trace artifacts first; then use browser traversal if DDGS extraction cannot reach the needed page state.",
                True,
            )
        )

    if "exhaustive_list" in stress_profiles and signals["domain_diversity"] < 3:
        gaps.append(
            make_gap(
                "coverage_gap",
                "medium",
                "The task asks for broad or exhaustive coverage, but the dossier has narrow domain or topic coverage.",
                {
                    "domain_diversity": signals["domain_diversity"],
                    "topic_cluster_count": signals["topic_cluster_count"],
                },
                "Add source-surface and contrastive facets before claiming completeness.",
                False,
            )
        )

    if "deep_research_report" in stress_profiles and signals["pass_count"] < 4:
        gaps.append(
            make_gap(
                "stopping_gap",
                "medium",
                "The task needs a stopping criterion; the dossier does not yet show whether further search is low-yield.",
                {"pass_count": signals["pass_count"]},
                "Run a broader query graph and compare pass-yield before final synthesis.",
                False,
            )
        )

    if "multimodal_browsing" in stress_profiles:
        gaps.append(
            make_gap(
                "modality_gap",
                "medium",
                "The task may rely on images, tables, video, UI layout, or PDFs beyond text extraction.",
                {"stress_profile": "multimodal_browsing"},
                "Use multimodal or browser inspection for the specific sources named in the packet.",
                True,
            )
        )

    if "enterprise_or_private_corpus" in stress_profiles:
        gaps.append(
            make_gap(
                "corpus_boundary_gap",
                "high",
                "The task may require private, logged-in, enterprise, or connector-backed data outside public web search.",
                {"stress_profile": "enterprise_or_private_corpus"},
                "Use the appropriate private-corpus or connector tool; public DDGS can only provide outside context.",
                True,
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


def budget_settings(args: argparse.Namespace) -> dict[str, int]:
    """Resolve compact packet budget settings."""
    defaults = {
        "tiny": {
            "top_sources": 3,
            "top_harvest": 5,
            "top_unpromoted": 3,
            "excerpt_chars": 160,
        },
        "normal": {
            "top_sources": 5,
            "top_harvest": 10,
            "top_unpromoted": 5,
            "excerpt_chars": 280,
        },
        "deep": {
            "top_sources": 8,
            "top_harvest": 20,
            "top_unpromoted": 8,
            "excerpt_chars": 500,
        },
    }[args.budget]
    if args.top_sources:
        defaults["top_sources"] = args.top_sources
    if args.top_harvest:
        defaults["top_harvest"] = args.top_harvest
    if args.top_unpromoted:
        defaults["top_unpromoted"] = args.top_unpromoted
    if args.excerpt_chars:
        defaults["excerpt_chars"] = args.excerpt_chars
    return defaults


def derive_agent_status(
    signals: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether the packet is enough for synthesis."""
    high_gaps = [gap for gap in gaps if gap["severity"] == "high"]
    if high_gaps:
        next_action = "run_refinement"
        gap_types = {gap["gap_type"] for gap in high_gaps}
        if "corpus_boundary_gap" in gap_types:
            next_action = "private_corpus_needed"
        elif "traversal_gap" in gap_types:
            next_action = "browser_traversal"
        elif "modality_gap" in gap_types:
            next_action = "multimodal_inspection"
        elif "extraction_gap" in gap_types:
            next_action = "extract_more"
        return {
            "sufficient": False,
            "confidence": "provisional",
            "next_action": next_action,
            "why": ", ".join(gap["gap_type"] for gap in high_gaps[:3]) + " remain",
        }

    if signals.get("requires_more_search"):
        return {
            "sufficient": False,
            "confidence": "partial",
            "next_action": "run_refinement",
            "why": "sufficiency signals still require more search",
        }

    confidence = (
        "verified"
        if signals.get("extracted_evidence_count", 0) > 0
        else "source_list_only"
    )
    return {
        "sufficient": True,
        "confidence": confidence,
        "next_action": "final_synthesis",
        "why": "no material high-severity retrieval gap remains",
    }


def compact_query_strategy(query_strategy: dict[str, Any]) -> dict[str, Any]:
    """Compress query strategy into a packet-sized summary."""
    facets = query_strategy.get("query_facets") or []
    anchors = [
        item["query"]
        for item in facets
        if item.get("exploration_anchor") and not item.get("memory_seeded")
    ]
    warnings = [item["type"] for item in query_strategy.get("strategy_warnings") or []]
    return {
        "stress_profiles": query_strategy.get("stress_profiles") or [],
        "grounding": "task_semantic" if anchors else "ungrounded",
        "exploration_anchors": anchors[:5],
        "memory_seeded_entities": query_strategy.get("hypothesis_entities") or [],
        "user_entities": query_strategy.get("user_entities") or [],
        "warnings": warnings,
    }


def compact_gaps(gaps: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Keep only decision-relevant gap fields."""
    priority = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(gaps, key=lambda gap: priority.get(gap["severity"], 9))
    return [
        {
            "type": gap["gap_type"],
            "severity": gap["severity"],
            "next_move": gap["preferred_ddgs_move"],
            "native_search_candidate": gap["native_search_candidate"],
        }
        for gap in ordered[:limit]
    ]


def select_agent_top_sources(
    recommended_reads: dict[str, dict[str, Any] | None],
    clusters: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select deduped top sources for compact agent reasoning."""
    cluster_lookup = {cluster["cluster_id"]: cluster for cluster in clusters}
    results = []
    seen = set()
    counter = 1
    for role, candidate in recommended_reads.items():
        if not candidate:
            continue
        url = candidate.get("url") or ""
        cluster_id = candidate["cluster_id"]
        key = url or cluster_id
        if key in seen:
            continue
        seen.add(key)
        cluster = cluster_lookup.get(cluster_id, {})
        results.append(
            {
                "source_id": f"S{counter}",
                "role": role,
                "title": candidate["title"],
                "url": url,
                "why": candidate["selection_reason"],
                "signals": candidate["signal_scores"],
                "trace": {
                    "cluster_id": cluster_id,
                    "pass_ids": candidate.get("pass_ids", []),
                    "query_facets": candidate.get("query_facets")
                    or cluster.get("query_facets", []),
                },
            }
        )
        counter += 1
        if len(results) >= limit:
            break
    return results


def summarize_extracted_evidence(
    extracted_pages: list[dict[str, Any]],
    top_sources: list[dict[str, Any]],
    *,
    excerpt_chars: int,
) -> list[dict[str, Any]]:
    """Attach tiny extraction notes for selected sources."""
    source_by_cluster = {
        source["trace"]["cluster_id"]: source["source_id"] for source in top_sources
    }
    notes = []
    for page in extracted_pages:
        source_id = source_by_cluster.get(page["cluster_id"])
        if not source_id:
            continue
        note = {
            "source_id": source_id,
            "cluster_id": page["cluster_id"],
            "url": page["url"],
        }
        if "error" in page:
            note["status"] = "error"
            note["error"] = page["error"][:excerpt_chars]
        else:
            note["status"] = "extracted"
            note["excerpt"] = page.get("preview", "").replace("\n", " ")[:excerpt_chars]
        notes.append(note)
    return notes


def build_next_query_suggestions(
    dossier: dict[str, Any], limit: int = 5
) -> list[dict[str, str]]:
    """Suggest small, faceted next DDGS moves."""
    suggestions = []
    query = dossier["intent"]["query"]
    gaps = {gap["gap_type"] for gap in dossier.get("open_gaps", [])}
    official_domains = dossier["intent"].get("official_domains") or []
    if "query_grounding_gap" in gaps:
        suggestions.append({"facet": "task_anchor", "query": query})
    if "authority_gap" in gaps and official_domains:
        for domain in official_domains[:2]:
            suggestions.append(
                {
                    "facet": "source_surface",
                    "query": f"site:{normalize_domain(domain)} {query}",
                }
            )
    if "freshness_gap" in gaps and "latest" not in query.lower():
        suggestions.append({"facet": "recency_probe", "query": f"{query} latest"})
    if "coverage_gap" in gaps or "diversity_gap" in gaps:
        suggestions.append(
            {"facet": "contrastive", "query": f"{query} alternatives comparison"}
        )

    for item in (dossier.get("candidate_audit", {}).get("unpromoted_candidates") or [])[
        :2
    ]:
        suggestions.append(
            {"facet": "harvest_refinement", "query": f"{item['name']} {query}"}
        )
        if len(suggestions) >= limit:
            return suggestions[:limit]

    for item in dossier.get("harvest_candidates", [])[:3]:
        if item.get("risk_flags") and not item.get("support", {}).get(
            "official_support"
        ):
            continue
        suggestions.append(
            {"facet": "harvest_refinement", "query": f"{item['term']} {query}"}
        )
        if len(suggestions) >= limit:
            break
    return suggestions[:limit]


def compact_unpromoted_candidates(
    candidate_audit: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep unpromoted candidate audit compact for agent stdout."""
    rows = []
    for item in (candidate_audit.get("unpromoted_candidates") or [])[:limit]:
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "name": item["name"],
                "kind": item["kind"],
                "why_notable": item["why_notable"],
                "source_surfaces": item["source_surfaces"],
                "promotion_score": item["scores"]["promotion_score"],
                "next_move": "Inspect before final synthesis; include, exclude with reason, or refine.",
                "trace": item["trace"],
            }
        )
    return rows


def build_agent_packet(
    dossier: dict[str, Any],
    trace_artifacts: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a compact decision packet for agent stdout."""
    budget = budget_settings(args)
    top_sources = select_agent_top_sources(
        dossier["recommended_reads"],
        dossier["clusters"],
        limit=budget["top_sources"],
    )
    evidence_notes = summarize_extracted_evidence(
        dossier["extracted_pages"],
        top_sources,
        excerpt_chars=budget["excerpt_chars"],
    )
    return {
        "run_id": dossier["run_meta"]["run_id"],
        "status": derive_agent_status(
            dossier["sufficiency_signals"], dossier["open_gaps"]
        ),
        "query_strategy": compact_query_strategy(dossier["query_strategy"]),
        "top_sources": top_sources,
        "evidence_notes": evidence_notes,
        "open_gaps": compact_gaps(dossier["open_gaps"]),
        "harvest_candidates": dossier["harvest_candidates"][: budget["top_harvest"]],
        "unpromoted_candidates": compact_unpromoted_candidates(
            dossier.get("candidate_audit", {}),
            limit=budget["top_unpromoted"],
        ),
        "next_queries": build_next_query_suggestions(dossier),
        "trace": trace_artifacts,
    }


def make_run_id(query: str) -> str:
    """Build a stable-enough run id for trace artifacts."""
    seed = f"{now_iso()}:{query}"
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + stable_id(seed)[:8]


def prepare_run_dir(args: argparse.Namespace, run_id: str) -> Path:
    """Create the trace artifact directory."""
    path = Path(args.run_dir) if args.run_dir else Path("output") / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL trace rows."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_replay(path: Path, argv: list[str]) -> None:
    """Write a replayable command."""
    path.write_text("#!/usr/bin/env bash\n" + shlex.join(argv) + "\n", encoding="utf-8")
    path.chmod(0o755)


def trace_artifact_paths(run_dir: Path) -> dict[str, str]:
    """Return trace artifact paths."""
    return {
        "agent_packet": str(run_dir / "agent_packet.json"),
        "full_json": str(run_dir / "full_dossier.json"),
        "raw_hits": str(run_dir / "raw_hits.jsonl"),
        "normalized_hits": str(run_dir / "normalized_hits.jsonl"),
        "extracts": str(run_dir / "extracts.jsonl"),
        "replay": str(run_dir / "replay.sh"),
    }


def write_trace_artifacts(
    trace_artifacts: dict[str, str],
    dossier: dict[str, Any],
    agent_packet: dict[str, Any],
    raw_hits: list[dict[str, Any]],
    normalized_results: list[dict[str, Any]],
    extracted_pages: list[dict[str, Any]],
    argv: list[str],
) -> None:
    """Persist full audit artifacts for selective inspection."""
    Path(trace_artifacts["agent_packet"]).write_text(
        json.dumps(agent_packet, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(trace_artifacts["full_json"]).write_text(
        json.dumps(dossier, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_jsonl(Path(trace_artifacts["raw_hits"]), raw_hits)
    write_jsonl(Path(trace_artifacts["normalized_hits"]), normalized_results)
    write_jsonl(Path(trace_artifacts["extracts"]), extracted_pages)
    write_replay(Path(trace_artifacts["replay"]), argv)


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
        "## Query Strategy",
        f"- Stress profiles: {', '.join(dossier['query_strategy'].get('stress_profiles') or []) or 'none'}",
        f"- Hypothesis entities: {', '.join(dossier['query_strategy'].get('hypothesis_entities') or []) or 'none'}",
        f"- User entities: {', '.join(dossier['query_strategy'].get('user_entities') or []) or 'none'}",
        "",
        "## Query Pack",
        f"- Canonical query: {dossier['query_pack']['canonical_query']}",
    ]
    for facet in dossier["query_pack"]["variants"]:
        lines.append(
            f"- Facet [{facet['facet']} / {facet['label']}]: {facet['query']} "
            f"(memory_seeded={facet['memory_seeded']}, exploration_anchor={facet['exploration_anchor']})"
        )

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
                f"- Query facets: {', '.join(cluster.get('query_facets', [])) or 'none'}",
                f"- Passes: {', '.join(cluster['pass_ids']) or 'none'}",
                f"- RRF fusion: {cluster['rank_fusion_score']}",
                f"- Signals: authority={signals['authority']}, answerability={signals['answerability']}, freshness={signals['freshness']}, corroboration={signals['corroboration']}, extractability={signals['extractability']}",
            ]
        )
        for support in cluster["supporting_results"][:4]:
            lines.append(
                f"- Support: {support['pass_id']} rank={support['rank_in_pass']} label={support['query_label']} facet={support.get('query_facet', 'unknown')} category={support['category']}"
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
                f"- Query facets: {', '.join(candidate.get('query_facets', [])) or 'none'}",
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
                lines.append(
                    f"- Preview: {page['preview'].replace(chr(10), ' ')[:800]}"
                )
            lines.append("")

    lines.append("## Harvest Candidates")
    if not dossier.get("harvest_candidates"):
        lines.append("- none")
        lines.append("")
    else:
        for item in dossier["harvest_candidates"][:10]:
            lines.append(
                f"- {item['term']} [{item['kind']}]: clusters={item['support']['cluster_count']} domains={item['support']['domain_count']} official={item['support']['official_support']} risks={', '.join(item['risk_flags']) or 'none'}"
            )
        lines.append("")

    lines.append("## Candidate Promotion Audit")
    unpromoted = dossier.get("candidate_audit", {}).get("unpromoted_candidates") or []
    if not unpromoted:
        lines.append("- none")
        lines.append("")
    else:
        for item in unpromoted[:10]:
            scores = item["scores"]
            lines.extend(
                [
                    f"- {item['name']} [{item['kind']}]: promotion={scores['promotion_score']} task_fit={scores['task_fit']} specificity={scores['specificity_fit']} surfaces={', '.join(item['source_surfaces']) or 'none'}",
                    f"  - Reason: {item['why_notable']}",
                    f"  - Trace clusters: {', '.join(item['trace']['cluster_ids']) or 'none'}",
                ]
            )
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

    query_strategy = build_query_strategy(args)
    query_pack = build_query_pack(args, query_strategy)
    strategy_warnings = build_query_strategy_warnings(
        query_pack,
        query_strategy.get("stress_profiles") or [],
        args,
        query_strategy.get("hypothesis_entities") or [],
    )
    query_strategy["query_facets"] = [asdict(facet) for facet in query_pack]
    query_strategy["strategy_warnings"] = strategy_warnings

    passes = build_passes(args, query_pack)
    official_domains = {
        normalized
        for normalized in (normalize_domain(domain) for domain in args.official_domain)
        if normalized
    }

    pass_records = []
    raw_hits = []
    normalized_results = []

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

            for rank_in_pass, raw in enumerate(raw_results, start=1):
                raw_hits.append(
                    {
                        "pass": asdict(plan),
                        "rank_in_pass": rank_in_pass,
                        "raw": raw,
                    }
                )
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
        recommended_reads = build_recommended_reads(
            clusters, args.authority, args.freshness
        )
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
    harvest_candidates = build_harvest_candidates(clusters, query=args.query)
    candidate_audit = build_candidate_audit(
        clusters,
        recommended_reads,
        query_strategy,
        query=args.query,
        freshness=args.freshness,
    )
    gaps = build_open_gaps(
        signals,
        recommended_reads,
        extracted_pages,
        args.authority,
        args.freshness,
        strategy_warnings=strategy_warnings,
        harvest_candidates=harvest_candidates,
        candidate_audit=candidate_audit,
        stress_profiles=query_strategy.get("stress_profiles") or [],
    )
    next_options = build_next_options(gaps, args.authority)
    run_id = make_run_id(args.query)
    run_dir = prepare_run_dir(args, run_id) if args.agent or args.run_dir else None
    trace_artifacts = trace_artifact_paths(run_dir) if run_dir else {}
    stdout_format = "agent_packet" if args.agent else args.stdout_format

    dossier = {
        "run_meta": {
            "created_at": now_iso(),
            "run_id": run_id,
            "stdout_format": stdout_format,
            "harness_profile": "deterministic_multi_signal",
            "agent_mode": args.agent,
        },
        "intent": {
            "query": args.query,
            "freshness": args.freshness,
            "authority": args.authority,
            "categories": parse_categories(args.categories, args.freshness),
            "region": args.region,
            "safesearch": args.safesearch,
            "official_domains": args.official_domain,
            "stress_profiles": query_strategy.get("stress_profiles") or [],
        },
        "query_strategy": query_strategy,
        "query_pack": {
            "canonical_query": args.query,
            "variants": [asdict(facet) for facet in query_pack],
        },
        "passes": pass_records,
        "pass_yield_summary": pass_yield_summary,
        "clusters": clusters,
        "recommended_reads": recommended_reads,
        "harvest_candidates": harvest_candidates,
        "candidate_audit": candidate_audit,
        "extracted_pages": extracted_pages,
        "sufficiency_signals": signals,
        "open_gaps": gaps,
        "next_options": next_options,
        "trace_artifacts": trace_artifacts,
    }
    agent_packet = build_agent_packet(dossier, trace_artifacts, args)
    dossier["agent_packet"] = agent_packet

    markdown = render_markdown(dossier)
    json_text = json.dumps(dossier, indent=2, ensure_ascii=False)

    if trace_artifacts:
        write_trace_artifacts(
            trace_artifacts,
            dossier,
            agent_packet,
            raw_hits,
            normalized_results,
            extracted_pages,
            sys.argv,
        )

    write_optional(args.write_markdown, markdown)
    write_optional(args.write_json, json_text)

    if stdout_format == "agent_packet":
        print(json.dumps(agent_packet, indent=2, ensure_ascii=False))
    elif stdout_format == "json":
        print(json_text)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
