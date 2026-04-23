---
name: ddgs-search-harness
description: DDGS-first web search harness for research, verification, source comparison, and current-information tasks. Use when Codex should avoid starting from memory-driven narrowing, default to ddgs for discovery and page extraction, and organize findings into a layered dossier before deciding whether any additional native web search is needed.
---

# DDGS Search Harness

Use this skill to make retrieval explicit and reproducible.

Start from `ddgs`, not from a remembered answer that later gets lightly verified.
Treat native `web_search` as optional gap-filling that the agent may choose later, not as the default first move.

## Default posture

- Use `ddgs` first for discovery.
- Stay in the DDGS lane first by using `ddgs.extract()` on the strongest candidates.
- Surface retrieval state as deterministic harness output.
- Let the agent decide whether any additional native `web_search` is justified.

## Quick start

Prefer the helper script for anything beyond a tiny one-off lookup:

```bash
python3 scripts/search_dossier.py "OpenAI Responses API" \
  --freshness current \
  --authority prefer_official \
  --official-domain developers.openai.com \
  --official-domain platform.openai.com
```

Use `--stdout-format json` if a machine-readable dossier is more useful than the Markdown reading surface.

## Workflow

### 1. Frame the search contract

Before searching, make the retrieval contract explicit:

- `freshness`
  Use `evergreen`, `current`, or `breaking`.
- `authority`
  Use `any`, `prefer_official`, or `official_only`.
- `categories`
  Default to `text` for evergreen questions.
  Default to `text,news` for current or breaking questions.
- `official domains`
  Add them when the question clearly has first-party sources.

Do not overfit this framing.
Its purpose is to shape the DDGS query pack and make the dossier legible.

### 2. Build a DDGS-first query pack

Start from a canonical query, then expand only as needed:

- canonical query
- recency-biased variant for `current` or `breaking`
- official-hint variant for `prefer_official` or `official_only`
- explicit `site:` variants for each official domain
- user-supplied variants when the wording itself may hide relevant results

Do not generate a huge query fanout by default.
The goal is breadth with control, not volume for its own sake.

### 3. Run DDGS passes before reasoning from memory

Use the helper script unless the task is truly tiny:

```bash
python3 scripts/search_dossier.py "query here"
```

Manual DDGS use is acceptable for very small tasks:

```python
from ddgs import DDGS

with DDGS() as ddgs:
    text_results = ddgs.text("query", max_results=5)
    page = ddgs.extract("https://example.com", fmt="text_plain")
```

If the question is time-sensitive, prefer:

- `news()` plus `text()`
- tighter `timelimit`
- explicit official domains when relevant

### 4. Read the dossier in order

The helper script outputs a layered dossier.
Read it in this order:

1. `Search Intent`
2. `Query Pack`
3. `DDGS Passes`
4. `Pass Yield Summary`
5. `Candidate Clusters`
6. `Recommended Reads`
7. `Extracted Evidence`
8. `Sufficiency Signals`
9. `Open Gaps`
10. `Next Options`

Use JSON as the truth surface.
Use Markdown as the reading surface.

For detailed field descriptions, read `references/dossier-schema.md`.

### 5. Stay in the DDGS lane first

If the current dossier is still weak, prefer these moves first:

- add official domains
- add one or two targeted query variants
- tighten recency
- increase `--extract-top-k`
- rerun DDGS with a better category mix

Do not switch tools just because the first pass is imperfect.

### 6. Decide whether native web search is necessary

This skill does not hardcode escalation thresholds.

Use the dossier state to decide.
Native `web_search` may be justified when:

- the dossier still lacks any official candidate for an official-source question
- the dossier lacks recent evidence for a current or breaking question
- DDGS candidates are relevant but still not directly answering the question
- DDGS extraction failed on the strongest candidates
- there is obvious cross-source conflict that needs additional verification

Even then, treat native search as optional supplementation, not as a replacement for the DDGS dossier.

## Interpretation rules

### Candidate clusters

Clusters are the main reading object.
They aggregate repeated or closely related hits so the agent can reason about:

- officialness
- answerability
- recency
- corroboration across passes
- extractability
- document family and topic-cluster support

Do not synthesize from raw hit order alone if the clusters tell a clearer story.
Treat each cluster as one canonical page with an added semantic layer, not as a loose bag of related URLs.

### Ranking and fusion

The harness does not trust raw DDGS list order by itself.
It keeps per-pass rank provenance and uses an RRF-style fusion signal across repeated hits.

It also keeps separate signal axes instead of collapsing everything into one opaque score:

- `authority`
- `answerability`
- `freshness`
- `corroboration`
- `extractability`

The agent should reason over these axes, not just over whichever URL appears first.

### Recommended reads

`recommended_reads` are explicit role slots, not generic top hits.

Current slots are:

- `official_anchor`
- `best_direct_answer`
- `fresh_update`
- `background_context`
- `alternate_view`

These slots are produced by the harness.
`official_anchor` specifically prefers reference-like official URLs when the query looks API-shaped.
`fresh_update` now uses a stricter topic guard so a merely recent but weakly aligned news hit does not occupy the slot too early.
The agent may override them when task context demands it, but should start from them.

### Pass yield

Each pass records marginal value, not only result count.

Read the `yield_analysis` fields to understand:

- how many clusters a pass touched
- how many clusters it introduced for the first time
- how much official, recent, or extractable novelty it added
- how redundant it was relative to earlier passes

Use this to decide whether another DDGS pass is likely to help.

### Sufficiency signals

These signals expose retrieval state, not mandatory routing:

- `official_cluster_count`
- `recent_cluster_count`
- `direct_cluster_count`
- `corroborated_cluster_count`
- `official_coverage`
- `recency_coverage`
- `direct_answer_coverage`
- `domain_diversity`
- `extracted_evidence_count`
- `requires_more_search`

Treat them as visibility aids for judgment.

### Open gaps

`open_gaps` are typed mechanical gaps, not free-form commentary.
Each gap includes:

- a `gap_type`
- a `severity`
- evidence explaining why the gap exists
- a preferred DDGS move
- whether native search is a reasonable candidate if DDGS still comes up short

Use them to decide what to do next.
Do not hide them in the final answer if they materially affect confidence.

## Output discipline

When answering from this skill's results:

- cite URLs you actually relied on
- distinguish `verified`, `provisional`, and `conflicted` claims when needed
- convert relative time into absolute dates when that matters
- prefer content extracted from pages over snippets from result lists

## Script notes

`scripts/search_dossier.py` supports:

- DDGS-first pass planning
- result normalization across categories
- RRF-style pass fusion
- candidate clustering with signal axes
- document-family and topic-cluster metadata
- explicit role-slot selection
- pass-yield analysis
- typed gap generation
- `ddgs.extract()` on top candidates
- dossier generation in Markdown or JSON

If `ddgs` is not importable:

- install it normally with `pip install ddgs`
- or set `DDGS_IMPORT_PATH` to a local ddgs checkout

The script also auto-detects a sibling `ddgs` repository when present.

## Files

- `scripts/search_dossier.py`
  Primary helper for building the layered dossier.
- `references/dossier-schema.md`
  Detailed schema and interpretation guide.
