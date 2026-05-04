import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search_dossier", ROOT / "scripts" / "search_dossier.py"
)
search_dossier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(search_dossier)


def args(**overrides):
    defaults = {
        "query": "latest instance segmentation model",
        "freshness": "current",
        "authority": "prefer_official",
        "categories": "",
        "backend": "auto",
        "region": "us-en",
        "safesearch": "moderate",
        "timelimit": "",
        "max_results_per_pass": 6,
        "extract_top_k": 3,
        "extract_format": "text_plain",
        "variant": [],
        "official_domain": ["paperswithcode.com", "arxiv.org"],
        "agent": True,
        "query_plan": "",
        "query_facet": [],
        "hypothesis_entity": ["SAM2"],
        "user_entity": [],
        "stress_profile": ["current_research"],
        "budget": "normal",
        "top_sources": 0,
        "top_harvest": 0,
        "top_unpromoted": 0,
        "excerpt_chars": 0,
        "run_dir": "",
        "stdout_format": "markdown",
        "write_json": "",
        "write_markdown": "",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_hypothesis_entity_is_not_exploration_anchor():
    parsed = args()
    strategy = search_dossier.build_query_strategy(parsed)
    facets = search_dossier.build_query_pack(parsed, strategy)

    hypothesis = [facet for facet in facets if facet.facet == "hypothesis_entity"]
    assert hypothesis
    assert all(facet.memory_seeded for facet in hypothesis)
    assert not any(facet.exploration_anchor for facet in hypothesis)


def test_recency_and_source_surface_derive_from_task_anchor_not_hypothesis():
    parsed = args()
    strategy = search_dossier.build_query_strategy(parsed)
    facets = search_dossier.build_query_pack(parsed, strategy)

    recency_queries = [
        facet.query for facet in facets if facet.facet == "recency_probe"
    ]
    source_queries = [
        facet.query for facet in facets if facet.facet == "source_surface"
    ]

    assert recency_queries
    assert source_queries
    assert all("SAM2" not in query for query in recency_queries)
    assert all("SAM2" not in query for query in source_queries)
    assert any(
        facet.facet == "task_anchor" and facet.exploration_anchor for facet in facets
    )


def test_entity_lock_in_gap_when_memory_seeded_majority():
    parsed = args(
        query_facet=[
            "hypothesis_entity:SAM2 instance segmentation",
            "hypothesis_entity:SAM3 instance segmentation",
        ],
        hypothesis_entity=[],
        official_domain=[],
        freshness="evergreen",
        authority="any",
    )
    strategy = search_dossier.build_query_strategy(parsed)
    facets = [
        search_dossier.QueryFacet(
            facet="hypothesis_entity",
            query="SAM2 instance segmentation",
            label="h1",
            memory_seeded=True,
            exploration_anchor=False,
        ),
        search_dossier.QueryFacet(
            facet="hypothesis_entity",
            query="SAM3 instance segmentation",
            label="h2",
            memory_seeded=True,
            exploration_anchor=False,
        ),
    ]
    warnings = search_dossier.build_query_strategy_warnings(
        facets, strategy["stress_profiles"], parsed
    )
    assert {warning["type"] for warning in warnings} >= {
        "query_grounding_gap",
        "entity_lock_in_gap",
    }


def test_contaminated_anchor_warns_when_query_contains_hypothesis():
    parsed = args(query="SAM2 latest instance segmentation", hypothesis_entity=["SAM2"])
    strategy = search_dossier.build_query_strategy(parsed)
    facets = search_dossier.build_query_pack(parsed, strategy)
    warnings = search_dossier.build_query_strategy_warnings(
        facets,
        strategy["stress_profiles"],
        parsed,
        strategy["hypothesis_entities"],
    )

    warning_types = {warning["type"] for warning in warnings}
    assert "query_grounding_gap" in warning_types
    assert "entity_lock_in_gap" in warning_types


def test_agent_packet_omits_full_clusters_and_keeps_trace_pointers():
    parsed = args()
    dossier = {
        "run_meta": {"run_id": "run-test"},
        "intent": {"query": parsed.query, "official_domains": parsed.official_domain},
        "query_strategy": {
            "stress_profiles": ["current_research"],
            "query_facets": [
                {
                    "facet": "task_anchor",
                    "query": parsed.query,
                    "exploration_anchor": True,
                    "memory_seeded": False,
                }
            ],
            "hypothesis_entities": ["SAM2"],
            "user_entities": [],
            "strategy_warnings": [],
        },
        "recommended_reads": {
            "official_anchor": {
                "role": "official_anchor",
                "cluster_id": "cluster-1",
                "title": "Instance segmentation leaderboard",
                "url": "https://paperswithcode.com/task/instance-segmentation",
                "pass_ids": ["pass-1"],
                "query_facets": ["source_surface"],
                "signal_scores": {
                    "authority": 1.0,
                    "answerability": 0.8,
                    "freshness": 0.4,
                    "corroboration": 0.5,
                    "extractability": 0.85,
                },
                "selection_reason": "Best authority surface.",
            },
            "best_direct_answer": None,
            "fresh_update": None,
            "background_context": None,
            "alternate_view": None,
        },
        "clusters": [
            {
                "cluster_id": "cluster-1",
                "query_facets": ["source_surface"],
                "signal_scores": {
                    "authority": 1.0,
                    "answerability": 0.8,
                    "freshness": 0.4,
                    "corroboration": 0.5,
                    "extractability": 0.85,
                },
            }
        ],
        "extracted_pages": [
            {
                "cluster_id": "cluster-1",
                "url": "https://paperswithcode.com/task/instance-segmentation",
                "preview": "short extracted evidence",
            }
        ],
        "sufficiency_signals": {
            "requires_more_search": False,
            "extracted_evidence_count": 1,
        },
        "open_gaps": [],
        "harvest_candidates": [],
    }

    packet = search_dossier.build_agent_packet(
        dossier,
        {
            "full_json": "full.json",
            "raw_hits": "raw.jsonl",
            "extracts": "extracts.jsonl",
        },
        parsed,
    )
    encoded = json.dumps(packet)

    assert "clusters" not in packet
    assert "passes" not in packet
    assert len(encoded.encode("utf-8")) < 8000
    assert packet["top_sources"][0]["trace"]["cluster_id"] == "cluster-1"
    assert packet["trace"]["raw_hits"] == "raw.jsonl"


def test_budget_tiny_limits_sources_and_harvest():
    parsed = args(budget="tiny")
    settings = search_dossier.budget_settings(parsed)
    assert settings == {
        "top_sources": 3,
        "top_harvest": 5,
        "top_unpromoted": 3,
        "excerpt_chars": 160,
    }


def test_harvest_filters_query_terms_and_generic_noise():
    clusters = [
        {
            "cluster_id": "cluster-1",
            "signal_scores": {
                "authority": 0.0,
                "answerability": 0.8,
                "freshness": 0.0,
                "corroboration": 0.2,
                "extractability": 0.85,
            },
            "topic_alignment_score": 0.8,
            "topic_support_score": 0.5,
            "topic_terms": ["instance", "segmentation", "model", "yolov8n"],
            "titles": ["YOLOv8n segmentation result review"],
            "canonical_url": "https://example.com/yolov8n-segmentation",
            "root_domain": "example.com",
            "official": False,
            "document_family_kind": "docs",
        }
    ]

    harvest = search_dossier.build_harvest_candidates(
        clusters,
        query="latest instance segmentation model",
    )
    terms = {item["term"].lower() for item in harvest}
    assert "instance" not in terms
    assert "segmentation" not in terms
    assert "model" not in terms
    assert "result" not in terms
    assert "review" not in terms
    assert "yolov8n" in terms


def cluster(
    cluster_id,
    title,
    url,
    snippet,
    *,
    root_domain,
    official=False,
    answerability=0.75,
    freshness=0.4,
    query_facets=None,
):
    return {
        "cluster_id": cluster_id,
        "canonical_url": url,
        "domain": root_domain,
        "root_domain": root_domain,
        "document_family_kind": "page",
        "topic_terms": [],
        "titles": [title],
        "snippets": [snippet],
        "query_facets": query_facets or ["task_anchor"],
        "official": official,
        "topic_alignment_score": 0.8,
        "topic_support_score": 0.5,
        "signal_scores": {
            "authority": 1.0 if official else 0.4,
            "answerability": answerability,
            "freshness": freshness,
            "corroboration": 0.2,
            "extractability": 0.85,
        },
    }


def empty_recommended_reads():
    return {
        "official_anchor": None,
        "best_direct_answer": None,
        "fresh_update": None,
        "background_context": None,
        "alternate_view": None,
    }


def test_candidate_audit_merges_cross_surface_high_fit_candidate():
    parsed = args(
        query="open source any-to-any image generation editing model",
        official_domain=["github.com", "huggingface.co"],
        stress_profile=["current_research"],
    )
    strategy = search_dossier.build_query_strategy(parsed)
    strategy["query_facets"] = [
        {
            "facet": "task_anchor",
            "query": parsed.query,
            "exploration_anchor": True,
            "memory_seeded": False,
        }
    ]
    clusters = [
        cluster(
            "cluster-github",
            "OpenSenseNova/SenseNova-U1",
            "https://github.com/OpenSenseNova/SenseNova-U1",
            "Unified any-to-any image generation and editing project.",
            root_domain="github.com",
            official=True,
        ),
        cluster(
            "cluster-hf",
            "sensenova/SenseNova-U1-8B-MoT",
            "https://huggingface.co/sensenova/SenseNova-U1-8B-MoT",
            "Open-source model for any-to-any image generation and editing.",
            root_domain="huggingface.co",
            official=True,
        ),
    ]

    audit = search_dossier.build_candidate_audit(
        clusters,
        empty_recommended_reads(),
        strategy,
        query=parsed.query,
        freshness="current",
    )

    names = {item["name"] for item in audit["unpromoted_candidates"]}
    assert "sensenova/SenseNova-U1-8B-MoT" in names
    candidate = next(
        item
        for item in audit["unpromoted_candidates"]
        if item["name"] == "sensenova/SenseNova-U1-8B-MoT"
    )
    assert set(candidate["source_surfaces"]) >= {"repo", "model_or_dataset"}
    assert set(candidate["trace"]["cluster_ids"]) == {"cluster-github", "cluster-hf"}
    assert candidate["scores"]["promotion_score"] >= 0.48


def test_candidate_audit_does_not_promote_generic_query_phrase():
    parsed = args(query="text to image model", official_domain=[])
    strategy = search_dossier.build_query_strategy(parsed)
    strategy["query_facets"] = [
        {
            "facet": "task_anchor",
            "query": parsed.query,
            "exploration_anchor": True,
            "memory_seeded": False,
        }
    ]
    audit = search_dossier.build_candidate_audit(
        [
            cluster(
                "cluster-generic",
                "Text-to-image model overview",
                "https://example.com/text-to-image-model",
                "A broad overview of text-to-image model methods.",
                root_domain="example.com",
            )
        ],
        empty_recommended_reads(),
        strategy,
        query=parsed.query,
        freshness="evergreen",
    )

    assert audit["unpromoted_candidates"] == []


def test_unpromoted_candidates_surface_in_gaps_and_agent_packet():
    parsed = args()
    candidate_audit = {
        "candidate_families": [],
        "unpromoted_candidates": [
            {
                "candidate_id": "candidate-1",
                "name": "Project-X-1B",
                "kind": "entity_candidate",
                "why_notable": "High-fit retrieved candidate flagged by task-fit.",
                "source_surfaces": ["repo"],
                "scores": {
                    "promotion_score": 0.7,
                    "promoted": False,
                    "task_fit": 0.6,
                    "specificity_fit": 0.8,
                },
                "official_support": False,
                "trace": {
                    "cluster_ids": ["cluster-x"],
                    "query_facets": ["task_anchor"],
                },
            }
        ],
        "merge_warnings": [],
        "promotion_thresholds": {},
    }
    gaps = search_dossier.build_open_gaps(
        {
            "official_cluster_count": 0,
            "recent_cluster_count": 0,
            "supported_topic_cluster_count": 1,
            "direct_cluster_count": 1,
            "extracted_evidence_count": 1,
            "domain_diversity": 2,
            "alternate_view_coverage": True,
            "topic_cluster_count": 2,
            "pass_count": 2,
        },
        {
            **empty_recommended_reads(),
            "best_direct_answer": {"cluster_id": "cluster-y"},
            "fresh_update": {"cluster_id": "cluster-y"},
        },
        [{"cluster_id": "cluster-y", "preview": "evidence"}],
        "any",
        "current",
        candidate_audit=candidate_audit,
    )
    assert "unpromoted_candidate_gap" in {gap["gap_type"] for gap in gaps}

    dossier = {
        "run_meta": {"run_id": "run-test"},
        "intent": {"query": parsed.query, "official_domains": []},
        "query_strategy": {
            "stress_profiles": [],
            "query_facets": [],
            "hypothesis_entities": [],
            "user_entities": [],
            "strategy_warnings": [],
        },
        "recommended_reads": empty_recommended_reads(),
        "clusters": [],
        "extracted_pages": [],
        "sufficiency_signals": {
            "requires_more_search": False,
            "extracted_evidence_count": 1,
        },
        "open_gaps": gaps,
        "harvest_candidates": [],
        "candidate_audit": candidate_audit,
    }
    packet = search_dossier.build_agent_packet(dossier, {}, parsed)
    assert packet["unpromoted_candidates"][0]["name"] == "Project-X-1B"
