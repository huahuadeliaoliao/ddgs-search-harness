---
name: ddgs-search-harness
description: Agent-first DDGS search harness for exploration-first research, current-information discovery, source comparison, and hard web-search tasks. Use when Codex should derive search terms from task concepts instead of remembered candidate answers, quarantine hypothesis entities, run DDGS-first retrieval, return compact decision packets, and preserve full trace artifacts before deciding whether refinement or native web search is needed.
---

# DDGS Search Harness

Use this skill to make search strategy, retrieval state, and evidence trace explicit and reproducible.

Start from the user's task concepts, not from a remembered candidate answer that later gets lightly verified.
Use DDGS as the first retrieval lane.
Treat native `web_search` as optional gap-filling that the agent may choose later, not as the default first move.

The default output for agent use should be a compact decision packet, not a full human-readable report.
Full evidence must remain available through trace artifacts.

## Default Posture

- Derive search terms from task concepts before naming candidate answers.
- Use `ddgs` first for discovery.
- Use compact agent packets for routine reasoning.
- Preserve full JSON and raw-hit trace artifacts for audit and follow-up.
- Stay in the DDGS lane first by using `ddgs.extract()` on the strongest candidates.
- Let the agent decide whether additional DDGS refinement, browser traversal, multimodal inspection, private-corpus access, or native `web_search` is justified.

## Quick Start

Prefer the helper script with `--agent` for anything beyond a tiny one-off lookup:

```bash
python3 scripts/search_dossier.py "latest instance segmentation model" \
  --agent \
  --freshness current \
  --stress-profile current_research \
  --official-domain paperswithcode.com \
  --official-domain arxiv.org \
  --official-domain github.com \
  --hypothesis-entity SAM2
```

Use the compact agent packet from stdout for the next decision.
Inspect full trace artifacts only when the packet says evidence, extraction, coverage, or provenance needs closer review.

Use full Markdown only when a human reading surface is explicitly needed:

```bash
python3 scripts/search_dossier.py "query here" --stdout-format markdown
```

## Workflow

### 1. Classify the search stress profile

Before building queries, classify the task into one or more stress profiles:

- `single_hard_fact`: a hard-to-find fact or entity that may require multiple phrasings and source triangulation.
- `current_research`: a latest/current/best question where memory may be stale and recency matters.
- `web_traversal`: a question likely requiring navigation through multi-level websites, not just search result snippets.
- `exhaustive_list`: a question asking for all, most, best, current, or complete sets of items.
- `deep_research_report`: a broad research task requiring many branches, synthesis, and a stopping criterion.
- `multilingual_web`: a task where relevant sources may use non-English terms, regional terminology, or local platforms.
- `multimodal_browsing`: a task where images, tables, PDFs, videos, UI screenshots, or page layout may carry key evidence.
- `enterprise_or_private_corpus`: a task likely requiring private, logged-in, or enterprise data beyond public web search.

Use these profiles to decide which gaps matter.
Do not assume a simple search-result page is enough for hard browsing or deep-research tasks.

### 2. Build a concept map before searching

Before DDGS search, build a compact concept map:

- `task_concepts`: core objects, actions, capabilities, constraints, and domain.
- `neutral_terms`: task-level and domain-level terms that do not assume a remembered answer.
- `synonyms_and_related_terms`: alternate names, broader terms, narrower terms, and adjacent vocabulary.
- `source_surfaces`: likely high-signal evidence surfaces, such as official docs, papers, repositories, leaderboards, changelogs, standards, or local-language sources.
- `constraints`: freshness, geography, language, modality, benchmark, version, price, legal, safety, or other task constraints.
- `hypothesis_entities`: candidate entities remembered by the agent or suggested as uncertain possibilities.
- `user_entities`: named entities explicitly supplied by the user.

The canonical query must come from `task_concepts` or `neutral_terms`.
Remembered entities must be quarantined as `hypothesis_entities`.
They may be searched, but they must not define the search space unless the user explicitly asks about that entity.

### 3. Build an agent-first query plan

Use query facets instead of untyped query variants:

- `task_anchor`: the neutral canonical task query.
- `domain_broadening`: broader or adjacent domain vocabulary.
- `source_surface`: searches aimed at authoritative surfaces such as papers, repos, official docs, leaderboards, changelogs, or standards.
- `recency_probe`: current/latest/recent variants for time-sensitive tasks.
- `contrastive`: a query designed to find alternatives, criticism, failures, comparisons, or competing approaches.
- `hypothesis_entity`: a remembered or tentative entity, quarantined from the exploration anchor.
- `user_entity`: a named entity explicitly supplied by the user.
- `harvest_refinement`: a second-pass query based on terms surfaced by prior DDGS results.

Do not overuse operators early.
Operators such as `site:`, exact quotes, `filetype:`, minus, and date filters are refinement tools.
Use them after at least one neutral exploration anchor exists, unless the user explicitly requested a known official domain.

### 4. Run DDGS passes in agent mode

Use:

```bash
python3 scripts/search_dossier.py "task-level query" --agent
```

Prefer `--query-plan` for complex tasks and `--query-facet` for small additions.
Use `--hypothesis-entity` for remembered candidate answers.

Read stdout as an agent packet.
Do not paste or reason over the full dossier unless the compact packet points to a specific trace artifact that matters.

### 5. Use guarded result-vocabulary harvest

Treat harvested result terms like pseudo relevance feedback.

Harvested terms are useful when they solve vocabulary mismatch.
They are risky when the first result set drifted off-topic.

A harvested term should guide a second pass only when it is supported by strong topic alignment, repeated cluster support, source diversity, official/paper/repo/benchmark/changelog evidence, or direct extraction from a high-quality source.

Do not let one noisy snippet steer the search.
Do not treat a merely recent result as a fresh update unless it is materially on-topic.

### 6. Decide the next move from gaps

Use the compact packet's open gaps and next actions.

DDGS refinement is preferred first when the gap is query grounding, source diversity, freshness, authority surface, extraction, or topic drift.

Native `web_search` may be justified when DDGS still lacks official evidence, recent evidence, direct answerability, or extraction after targeted refinement.
Browser traversal may be needed for `web_traversal`.
Multimodal tools may be needed for `multimodal_browsing`.
Private or connector-backed tools may be needed for `enterprise_or_private_corpus`.

Do not hide material gaps in the final answer.

## Agent Output Discipline

The agent packet is the primary reasoning surface.

Use the packet to answer:

- Is the search strategy grounded in task concepts?
- Did any remembered entity contaminate the exploration anchor?
- Which sources should be read first?
- Which gaps materially affect confidence?
- Should the next move be DDGS refinement, extraction, native web search, browser traversal, multimodal inspection, private-corpus access, or final synthesis?

Use full trace artifacts only for selective inspection.
Do not load the full dossier into context unless the compact packet is insufficient.

Before final synthesis, inspect `unpromoted_candidates`, material `harvest_candidates`, and open gaps.
If a high-fit candidate was retrieved but not promoted, do one of:

- include it in the final synthesis
- explicitly explain why it is excluded
- run a targeted refinement before answering

Do not finalize from `top_sources` alone when candidate audit or coverage gaps remain.

For evidence-heavy answers, optionally run `scripts/synthesis_audit.py` against a draft answer and the full dossier.
Use the audit to catch unsupported claims and retrieved high-fit candidates that the draft omitted.

Traceability must be preserved through:

- `run_id`
- `source_id`
- `cluster_id`
- `pass_id`
- `query_facet`
- `query_label`
- source URL
- raw hit artifact path
- extracted evidence artifact path

## Interpretation Rules

### Candidate clusters

Clusters are the main full-dossier object.
They aggregate repeated or closely related hits so the agent can reason about officialness, answerability, recency, corroboration, extractability, document family, topic support, and query provenance.

Do not synthesize from raw hit order alone if the clusters tell a clearer story.

### Ranking and fusion

The harness does not trust raw DDGS order by itself.
It keeps per-pass rank provenance and uses an RRF-style fusion signal across repeated hits.

It also keeps separate signal axes:

- `authority`
- `answerability`
- `freshness`
- `corroboration`
- `extractability`

### Recommended reads

`recommended_reads` are explicit role slots, not generic top hits:

- `official_anchor`
- `best_direct_answer`
- `fresh_update`
- `background_context`
- `alternate_view`

The agent may override them when task context demands it, but should start from them.

### Open gaps

`open_gaps` are typed mechanical gaps, not free-form commentary.
Each gap includes a type, severity, evidence, preferred DDGS move, and whether native search could be considered later.

Do not hide material gaps in the final answer.

## Output Discipline

When answering from this skill's results:

- cite URLs you actually relied on
- distinguish `verified`, `provisional`, and `conflicted` claims when needed
- convert relative time into absolute dates when that matters
- prefer content extracted from pages over snippets from result lists

## Script Notes

`scripts/search_dossier.py` supports:

- concept-grounded DDGS pass planning
- query facets and hypothesis quarantine
- compact `--agent` packets
- run trace artifacts under `output/runs/<run_id>/`
- result normalization across categories
- RRF-style pass fusion
- candidate clustering with signal axes
- document-family and topic-cluster metadata
- explicit role-slot selection
- pass-yield analysis
- guarded harvest candidates
- candidate promotion audit for retrieved-but-not-promoted entities
- typed gap generation
- `ddgs.extract()` on top candidates
- optional draft-answer synthesis audit
- optional dossier generation in Markdown or JSON

If `ddgs` is not importable:

- install it normally with `pip install ddgs`
- or set `DDGS_IMPORT_PATH` to a local ddgs checkout

The script also auto-detects a sibling `ddgs` repository when present.

## References

- `references/query-strategy.md`: query grounding, facets, hypothesis quarantine, harvest discipline, and stress profiles.
- `references/agent-packet-schema.md`: compact packet shape, budgets, and trace requirements.
- `references/dossier-schema.md`: full dossier schema and interpretation guide.

## Files

- `scripts/search_dossier.py`: primary helper for building the layered dossier and compact packet.
- `references/`: detailed schema and strategy notes for selective loading.
