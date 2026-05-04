# Dossier Schema

`search_dossier.py` produces three complementary machine surfaces and one optional human surface:

- Agent packet is the compact decision surface for the current agent turn.
- Full JSON dossier is the structured truth surface for downstream reasoning and audit.
- Trace artifacts preserve raw hits, normalized hits, extracted evidence, and replay commands.
- Markdown is an optional human reading surface.

The design goal is an agent-first deterministic retrieval harness:

- the agent constructs a concept-grounded search strategy,
- the harness executes DDGS passes and records query provenance,
- the harness computes fusion, clusters, roles, harvest candidates, candidate promotion audit, gaps, and next actions,
- the agent decides whether to refine, inspect traces, escalate tools, or synthesize.

## Top-Level Fields

- `run_meta`
  Runtime metadata, run id, output mode, artifact paths, and harness profile.
- `intent`
  The search contract: query, freshness, authority, categories, region, safesearch, official domains, and stress profiles.
- `query_strategy`
  Concept map, query facets, hypothesis entities, user entities, and strategy warnings.
- `query_pack`
  Concrete DDGS query variants derived from the query strategy.
- `passes`
  Every executed DDGS pass, including query provenance, result count, and `yield_analysis`.
- `pass_yield_summary`
  A compact view of which passes added novelty and which were mostly redundant.
- `clusters`
  Aggregated candidate clusters built from normalized DDGS results.
- `recommended_reads`
  Explicit role slots chosen from the cluster set.
- `harvest_candidates`
  Terms, entities, benchmarks, repos, papers, or source surfaces surfaced by result clusters for possible refinement.
- `candidate_audit`
  Retrieved candidate families, high-fit unpromoted candidates, merge warnings, and promotion thresholds.
- `extracted_pages`
  Pages fetched through `ddgs.extract()` for content-level evidence.
- `sufficiency_signals`
  Compact signals describing the current state of coverage.
- `open_gaps`
  Typed missing evidence or strategy-risk records.
- `next_options`
  Suggested next retrieval or tool actions. These are advisory, not mandatory routing rules.
- `agent_packet`
  Optional compact packet intended for stdout and current-turn agent reasoning.
- `trace_artifacts`
  File paths for full dossier, raw hits, normalized hits, extracts, packet, and replay command.

## Query Strategy Fields

`query_strategy` describes why the harness searched the way it searched.

- `stress_profiles`
  One or more task stress profiles, such as `single_hard_fact`, `current_research`, `web_traversal`, `exhaustive_list`, `deep_research_report`, `multilingual_web`, `multimodal_browsing`, or `enterprise_or_private_corpus`.
- `concept_map`
  Task concepts, neutral terms, synonyms, broader/narrower terms, source surfaces, constraints, and hypothesis entities.
- `query_facets`
  Typed query inputs used to build DDGS passes.
- `hypothesis_entities`
  Remembered or tentative entities quarantined from the exploration anchor.
- `user_entities`
  Entities explicitly supplied by the user.
- `strategy_warnings`
  Query-grounding warnings such as entity lock-in, operator overfit, missing neutral anchor, or language-scope mismatch.

Each query facet includes:

- `facet`
- `query`
- `label`
- `source`
- `memory_seeded`
- `exploration_anchor`
- `operator_intensity`
- `notes`

## Candidate Audit Fields

`candidate_audit` is the retrieved-to-synthesis guardrail.
It looks for specific, task-fit candidates found in raw hits or clusters that did not reach `recommended_reads` or the compact top-source surface.

It must not hardcode domain candidate names.
Promotion signals are derived from the user query, query facets, concept map, source surface, result provenance, candidate specificity, and trace support.

Fields:

- `candidate_families`
  Alias-merged candidate entities with source surfaces, scores, surface forms, and trace pointers.
- `unpromoted_candidates`
  High-fit retrieved candidates that are not represented by promoted reading roles.
- `merge_warnings`
  Cross-surface candidate families that remain important to inspect before synthesis.
- `promotion_thresholds`
  Mechanical thresholds used by the deterministic audit.

Unpromoted candidates can create:

- `unpromoted_candidate_gap`
- `official_hit_not_promoted_gap`
- `entity_merge_gap`

## Synthesis Audit

`scripts/synthesis_audit.py` can audit a draft answer against a full dossier.

It performs two conservative checks:

- answer-to-evidence: claim-like sentences should have lexical support in clusters or extracted pages.
- evidence-to-answer: high-fit unpromoted candidates should be included, explicitly excluded, or refined before final synthesis.

This audit is deterministic and advisory.
It is designed to catch omissions and unsupported claims, not to replace agent judgment.

## Pass Fields

Each item in `passes` includes the original pass plan plus:

- `query_facet`
- `memory_seeded`
- `exploration_anchor`
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

## Cluster Fields

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
- `query_facets`
- `memory_seeded_support_count`
- `exploration_anchor_support_count`
- `queries`
- `result_count`
- `official`
- `rank_fusion_score`
- `signal_scores`
- `supporting_results`
- `snippets`

## Signal Scores

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

## Recommended-Read Roles

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
- `query_facets`
- `memory_seeded_support_count`
- `exploration_anchor_support_count`
- `signal_scores`
- `selection_reason`

## Agent Packet Fields

`agent_packet` is optimized for agent context, not human reading.

It should contain:

- `run_id`
- `status`
  Sufficiency, confidence, next action, and short reason.
- `query_strategy`
  Compact strategy summary and warnings.
- `top_sources`
  Small list of source summaries with `source_id`, role, title, URL, why selected, signals, and trace pointers.
- `evidence_notes`
  Short extracted snippets or extraction status for selected sources only.
- `open_gaps`
  Small list of material gaps and preferred next moves.
- `harvest_candidates`
  Small list of guarded candidate terms for refinement.
- `next_queries`
  Suggested next query facets, if another DDGS pass is useful.
- `trace`
  Paths to full dossier and raw artifacts.

The packet should avoid:

- full cluster lists,
- full pass records,
- long snippets,
- full extracted pages,
- raw hit payloads,
- Markdown prose sections.

## Sufficiency Signals

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

## Gap Fields

`open_gaps` is a list of typed objects.

Each gap contains:

- `gap_type`
- `severity`
- `reason`
- `evidence`
- `preferred_ddgs_move`
- `native_search_candidate`

Strategy gaps include:

- `query_grounding_gap`
- `entity_lock_in_gap`
- `operator_overfit_gap`
- `language_scope_gap`

Hard-search gaps include:

- `harvest_drift_risk`
- `traversal_gap`
- `coverage_gap`
- `stopping_gap`
- `modality_gap`
- `corpus_boundary_gap`

## Trace Artifacts

When `--agent` or `--run-dir` is used, the harness writes:

- `agent_packet.json`
- `full_dossier.json`
- `raw_hits.jsonl`
- `normalized_hits.jsonl`
- `extracts.jsonl`
- `replay.sh`

Every top source in the agent packet should include enough trace pointers to recover source URL, cluster id, pass ids, query facets, signal scores, raw hit record, and extracted evidence record when available.

## Escalation Semantics

This skill is DDGS-first.

Do not treat `open_gaps` or `requires_more_search` as automatic instructions to switch tools.
Use them to decide whether to:

1. rerun DDGS with better query or timelimit control,
2. add or tighten official-domain variants,
3. extract more DDGS candidates,
4. inspect trace artifacts,
5. use browser or multimodal tools for traversal/modality gaps,
6. or optionally supplement with native web search when the dossier still lacks the evidence needed.
