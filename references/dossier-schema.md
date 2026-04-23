# Dossier Schema

`search_dossier.py` produces two complementary views:

- JSON is the truth surface for downstream reasoning.
- Markdown is the reading surface for quick human and agent inspection.

The design goal is a deterministic retrieval harness:

- the harness computes search structure, fusion, roles, and typed gaps
- the agent decides what to do next with that structure

## Top-level fields

- `run_meta`
  Runtime metadata for the dossier generation pass.
- `intent`
  The search contract: query, freshness, authority, categories, region, and official domains.
- `query_pack`
  The canonical query and generated or user-supplied query variants.
- `passes`
  Every executed DDGS pass, including query provenance, result count, and `yield_analysis`.
- `pass_yield_summary`
  A compact view of which passes added novelty and which were mostly redundant.
- `clusters`
  Aggregated candidate clusters built from normalized DDGS results.
- `recommended_reads`
  Explicit role slots chosen from the cluster set.
- `extracted_pages`
  Pages fetched through `ddgs.extract()` for content-level evidence.
- `sufficiency_signals`
  Compact signals describing the current state of coverage.
- `open_gaps`
  Typed missing evidence records.
- `next_options`
  Suggested next retrieval actions. These are advisory, not mandatory routing rules.

## Pass fields

Each item in `passes` includes the original pass plan plus:

- `result_count`
- optional `error`
- `yield_analysis`

`yield_analysis` contains:

- `touched_cluster_count`
- `novel_cluster_count`
- `novel_official_count`
- `novel_fresh_count`
- `novel_extractable_count`
- `unique_domain_count`
- `mean_rank_in_pass`
- `redundancy_ratio`

Read these fields as marginal yield, not as model judgment.

## Cluster fields

Each cluster represents one canonical page plus a semantic grouping layer.

- `cluster_id`
- `canonical_url`
- `domain`
- `root_domain`
- `document_family`
- `document_family_size`
- `document_family_kind`
- `document_family_group`
- `document_family_group_size`
- `topic_terms`
- `topic_signature`
- `topic_cluster_key`
- `topic_cluster_size`
- `topic_cluster_official_count`
- `topic_cluster_domain_count`
- `topic_support_score`
- `topic_alignment_score`
- `titles`
- `categories`
- `sources`
- `pass_ids`
- `query_labels`
- `queries`
- `result_count`
- `official`
- `rank_fusion_score`
- `signal_scores`
- `supporting_results`
- `snippets`

## Signal scores

The harness keeps separate axes instead of compressing everything into one opaque score.

`signal_scores` contains:

- `authority`
  Confidence that the result belongs to an official source family.
- `answerability`
  How directly the result appears to answer the query.
- `freshness`
  Recency signal derived from DDGS-provided dates.
- `corroboration`
  How consistently the cluster reappears across passes, labels, and categories.
- `extractability`
  How likely `ddgs.extract()` is to return usable page content.

The cluster object also exposes semantic support outside `signal_scores`:

- `topic_support_score`
  How well the topic is supported across pages and domains.
- `topic_alignment_score`
  How strongly the page itself appears to be about the query topic, rather than only matching in snippets.

## Recommended-read roles

`recommended_reads` is a role-slot object, not a top-k list.

Current slots:

- `official_anchor`
- `best_direct_answer`
- `fresh_update`
- `background_context`
- `alternate_view`

Each populated role includes:

- `role`
- `cluster_id`
- `title`
- `url`
- `domain`
- `root_domain`
- `document_family`
- `document_family_kind`
- `topic_signature`
- `topic_cluster_key`
- `topic_support_score`
- `topic_alignment_score`
- `pass_ids`
- `signal_scores`
- `selection_reason`

## Sufficiency signals

The script exposes retrieval state, not hard routing decisions.

- `pass_count`
- `cluster_count`
- `domain_diversity`
- `topic_cluster_count`
- `supported_topic_cluster_count`
- `official_cluster_count`
- `recent_cluster_count`
- `direct_cluster_count`
- `corroborated_cluster_count`
- `official_coverage`
- `recency_coverage`
- `direct_answer_coverage`
- `alternate_view_coverage`
- `extracted_evidence_count`
- `requires_more_search`

## Gap fields

`open_gaps` is a list of typed objects.

Each gap contains:

- `gap_type`
- `severity`
- `reason`
- `evidence`
- `preferred_ddgs_move`
- `native_search_candidate`

This is the main handoff from harness to agent policy.

## Escalation semantics

This skill is `ddgs-first`.

Do not treat `open_gaps` or `requires_more_search` as automatic instructions to switch tools.
Use them to decide whether to:

1. rerun DDGS with better query or timelimit control,
2. add or tighten official-domain variants,
3. extract more DDGS candidates,
4. or optionally supplement with native web search when the dossier still lacks the evidence needed.

For current-information tasks, `fresh_update` is intentionally conservative:

- recent-but-off-topic results should fail the slot,
- strong topical alignment matters more than a raw news date,
- and a visible `freshness_gap` is preferable to a fabricated "latest update" claim.
