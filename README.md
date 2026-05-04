# DDGS Search Harness

Agent-first DDGS search harness for Codex that turns concept-grounded query plans into compact decision packets, full trace artifacts, ranked clusters, role slots, harvested refinement terms, typed gaps, and measured escalation options.

## Why this exists

Most agents search too late, too opaquely, or too narrowly:

- they start from remembered candidate answers,
- they use search to validate memory instead of exploring the task space,
- they treat one search page as enough evidence,
- they paste large retrieval dumps into context,
- and they escalate to stronger tools before making retrieval state explicit.

This repository packages a different default:

- start with task concepts,
- quarantine remembered entities as hypotheses,
- use DDGS first,
- return a compact agent packet for the next decision,
- preserve full evidence trace artifacts on disk,
- expose what is known and what is missing,
- and only then decide whether refinement, traversal, multimodal tools, private-corpus tools, or native search is justified.

## What the harness does

`scripts/search_dossier.py` builds a structured retrieval surface with:

- concept-grounded query facets and pass plans
- hypothesis-entity quarantine
- compact `--agent` packet output
- full trace artifacts for raw hits, normalized hits, extracts, and replay
- RRF-style pass fusion
- multi-axis cluster scoring
- document-family and topic-cluster metadata
- explicit recommended-read roles
- guarded harvest candidates for refinement
- candidate promotion audit for high-fit retrieved candidates that were not promoted
- `ddgs.extract()` previews for the strongest pages
- sufficiency signals, typed gaps, and next-step options

## Repository layout

- [SKILL.md](./SKILL.md): Codex skill instructions
- [agents/openai.yaml](./agents/openai.yaml): UI metadata for the skill
- [scripts/search_dossier.py](./scripts/search_dossier.py): deterministic DDGS dossier and packet builder
- [scripts/synthesis_audit.py](./scripts/synthesis_audit.py): optional draft-answer audit against a full dossier
- [references/query-strategy.md](./references/query-strategy.md): query grounding and search-strategy guide
- [references/agent-packet-schema.md](./references/agent-packet-schema.md): compact agent packet schema
- [references/dossier-schema.md](./references/dossier-schema.md): full schema notes

## Local use

Agent-first default:

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

Complex query plan:

```bash
python3 scripts/search_dossier.py "latest instance segmentation model" \
  --agent \
  --query-plan examples/query-plans/instance-segmentation.json
```

Human reading surface, when needed:

```bash
python3 scripts/search_dossier.py "OpenAI Responses API" \
  --stdout-format markdown \
  --write-json output/demo.json \
  --write-markdown output/demo.md
```

Audit a draft synthesis against a full dossier:

```bash
python3 scripts/synthesis_audit.py \
  --dossier output/runs/<run_id>/full_dossier.json \
  --answer draft_answer.md \
  --agent
```

Install as a local Codex skill by linking the repo into `~/.codex/skills`:

```bash
ln -s /absolute/path/to/ddgs-search-harness ~/.codex/skills/ddgs-search-harness
```

## Output model

The harness separates four layers:

- agent packet: compact decision surface for the current agent turn
- full dossier: structured truth surface for downstream reasoning and audit
- raw trace artifacts: raw hits, normalized hits, extracts, and replay command
- optional Markdown: human reading surface

The agent packet should be small enough to keep in context.
The full dossier and raw trace should remain on disk for selective inspection.

The harness keeps page clusters and semantic metadata separate:

- page clusters: one canonical page or deduped hit family
- semantic metadata: document family, family kind, topic cluster, support, alignment, query provenance, candidate promotion audit, and stress-profile gaps

## Design posture

- task concepts first, remembered candidates second
- DDGS first, native search second
- deterministic structure before free-form synthesis
- compact agent packet before full dossier inspection
- final synthesis should inspect unpromoted candidates before relying on top sources alone
- visible gaps instead of bluffing
- better to leave `fresh_update` empty than let off-topic recency noise occupy it

## License

[MIT](./LICENSE)
