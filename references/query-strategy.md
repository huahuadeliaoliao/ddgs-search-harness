# Query Strategy Guide

This guide defines how agents should construct DDGS search plans.

## Core Rule

The search plan must start from task concepts, not from remembered candidate answers.

Remembered entities may be useful, but they must be quarantined as `hypothesis_entity` facets.
A hypothesis entity may be searched, but it must not define the initial search space unless the user explicitly asks about that entity.

## Search Strategy Loop

Use this loop for non-trivial search tasks:

1. Classify the stress profile.
2. Build a concept map.
3. Run neutral task and domain anchors.
4. Inspect the agent packet.
5. Harvest result vocabulary with guardrails.
6. Run refinement facets when needed.
7. Inspect unpromoted candidates before synthesis.
8. Check gaps and decide whether to synthesize or escalate.

## Concept Map

A concept map should include:

- task concepts
- neutral terms
- synonyms and related terms
- broader and narrower terms
- source surfaces
- constraints
- hypothesis entities
- user entities

## Query Facets

Use these facets:

- `task_anchor`
- `domain_broadening`
- `source_surface`
- `recency_probe`
- `contrastive`
- `hypothesis_entity`
- `user_entity`
- `harvest_refinement`

## Hypothesis Quarantine

A remembered entity is a hypothesis, not an answer.

Bad:

```bash
python3 scripts/search_dossier.py "SAM2 latest instance segmentation"
```

Better:

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

## Operator Discipline

Operators are refinement tools.

Avoid starting with narrow operators unless the user explicitly asks for a source or domain.
Prefer at least one neutral task anchor before `site:`, exact quote, `filetype:`, minus, or date filters.

## Harvest Discipline

Harvested terms are candidate vocabulary, not confirmed facts.

A harvested term may guide refinement when it has:

- topic-aligned cluster support,
- repeated result support,
- source diversity,
- official, paper, repo, benchmark, or changelog support,
- or content-level extraction from a high-quality source.

A harvested term should be treated as risky when it is:

- found only in one weak snippet,
- recent but off-topic,
- from a low-quality aggregation page,
- unsupported by extractable evidence,
- or inconsistent with the task concept map.

## Candidate Promotion Audit

Before final synthesis, inspect `unpromoted_candidates`.

These are retrieved candidates that look specific and task-fit but were not promoted into top sources or recommended reads.
The audit is domain-neutral: it must not hardcode candidate names or task-specific boost keywords.

If a high-fit candidate was retrieved but not promoted, do one of:

- include it in the final synthesis,
- explicitly explain why it is excluded,
- or run targeted refinement before answering.

Do not finalize from top sources alone when an unpromoted-candidate or coverage gap remains.

For evidence-heavy answers, use `scripts/synthesis_audit.py` on a draft answer and the full dossier.
Treat the audit as a conservative warning layer for unsupported claims and omitted high-fit candidates.

## Stress Profiles

Use stress profiles to decide which gaps matter:

- `single_hard_fact`
- `current_research`
- `web_traversal`
- `exhaustive_list`
- `deep_research_report`
- `multilingual_web`
- `multimodal_browsing`
- `enterprise_or_private_corpus`
