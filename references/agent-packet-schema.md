# Agent Packet Schema

The agent packet is the default reasoning surface for agents.
It is designed to be compact, actionable, and traceable.

## Size Target

Default packet size should stay under 8 KB for normal searches.

Use budgets:

- `tiny`
  At most 3 sources, 5 harvest candidates, 3 unpromoted candidates, 160 character excerpts.
- `normal`
  At most 5 sources, 10 harvest candidates, 5 unpromoted candidates, 280 character excerpts.
- `deep`
  At most 8 sources, 20 harvest candidates, 8 unpromoted candidates, 500 character excerpts.

Full evidence belongs in trace artifacts, not stdout.

## Shape

```json
{
  "run_id": "20260503-abc123",
  "status": {
    "sufficient": false,
    "confidence": "provisional",
    "next_action": "run_refinement",
    "why": "freshness_gap and traversal_gap remain"
  },
  "query_strategy": {
    "stress_profiles": ["current_research"],
    "grounding": "task_semantic",
    "memory_seeded_entities": ["SAM2"],
    "warnings": ["hypothesis_entity_quarantined"]
  },
  "top_sources": [
    {
      "source_id": "S1",
      "role": "official_anchor",
      "title": "Example",
      "url": "https://example.com",
      "why": "official, direct, extractable",
      "signals": {
        "authority": 1.0,
        "answerability": 0.82,
        "freshness": 0.4,
        "corroboration": 0.5,
        "extractability": 0.85
      },
      "trace": {
        "cluster_id": "cluster-abc",
        "pass_ids": ["pass-3"],
        "query_facets": ["source_surface"]
      }
    }
  ],
  "open_gaps": [
    {
      "type": "query_grounding_gap",
      "severity": "high",
      "next_move": "add a neutral task-anchor query"
    }
  ],
  "harvest_candidates": [
    {
      "term": "SAM 3.1",
      "kind": "model_candidate",
      "support": {
        "cluster_count": 3,
        "domain_count": 2,
        "official_support": false
      },
      "risk": "needs_official_confirmation"
    }
  ],
  "unpromoted_candidates": [
    {
      "candidate_id": "candidate-abc",
      "name": "Project-X-1B",
      "kind": "entity_candidate",
      "why_notable": "High-fit retrieved candidate flagged by task-fit, authority-surface, specific-entity.",
      "source_surfaces": ["repo", "model_or_dataset"],
      "promotion_score": 0.76,
      "next_move": "Inspect before final synthesis; include, exclude with reason, or refine.",
      "trace": {
        "cluster_ids": ["cluster-abc"],
        "query_facets": ["task_anchor", "source_surface"]
      }
    }
  ],
  "next_queries": [
    {
      "facet": "source_surface",
      "query": "site:paperswithcode.com instance segmentation leaderboard 2026"
    }
  ],
  "trace": {
    "full_json": "output/runs/20260503-abc123/full_dossier.json",
    "raw_hits": "output/runs/20260503-abc123/raw_hits.jsonl",
    "extracts": "output/runs/20260503-abc123/extracts.jsonl",
    "replay": "output/runs/20260503-abc123/replay.sh"
  }
}
```

## Trace Rules

Every top source must include enough trace pointers to recover:

- source URL,
- cluster id,
- pass ids,
- query labels or query facets,
- signal scores,
- raw hit record,
- extracted evidence record when available.

Every unpromoted candidate must include enough trace pointers to recover the supporting clusters and query facets.

Before final synthesis, agents should inspect `unpromoted_candidates`.
If a high-fit retrieved candidate is omitted, the final answer should include it, explain why it was excluded, or run targeted refinement.

## Omitted From Packet

The packet should not include:

- full cluster lists,
- full pass records,
- long snippets,
- full extracted pages,
- raw hit payloads,
- Markdown prose sections.
