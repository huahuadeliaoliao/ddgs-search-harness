# DDGS Search Harness

DDGS-first search harness for Codex that turns raw search results into a deterministic dossier with ranked clusters, role slots, typed gaps, and measured escalation to native search.

## Why this exists

Most agents search too late or too opaquely:

- they narrow from memory first,
- they treat one search page as enough evidence,
- and they escalate to a stronger search tool before making retrieval state explicit.

This repository packages a different default:

- start with DDGS,
- fuse multiple controlled passes into one dossier,
- expose what is known and what is still missing,
- and only then decide whether more search is justified.

## What the harness does

`scripts/search_dossier.py` builds a structured retrieval surface with:

- a DDGS query pack and pass plan
- RRF-style pass fusion
- multi-axis cluster scoring
- document-family and topic-cluster metadata
- explicit recommended-read roles
- `ddgs.extract()` previews for the strongest pages
- sufficiency signals, typed gaps, and next-step options

The output is available as Markdown for reading and JSON for downstream agent reasoning.

## Repository layout

- [SKILL.md](./SKILL.md): Codex skill instructions
- [agents/openai.yaml](./agents/openai.yaml): UI metadata for the skill
- [scripts/search_dossier.py](./scripts/search_dossier.py): deterministic DDGS dossier builder
- [references/dossier-schema.md](./references/dossier-schema.md): field-level schema notes

## Local use

Run the helper directly:

```bash
python3 scripts/search_dossier.py "OpenAI Responses API" \
  --freshness current \
  --authority prefer_official \
  --official-domain developers.openai.com \
  --official-domain platform.openai.com \
  --official-domain openai.com
```

Install as a local Codex skill by linking the repo into `~/.codex/skills`:

```bash
ln -s /absolute/path/to/ddgs-search-harness ~/.codex/skills/ddgs-search-harness
```

## Output model

The harness keeps two layers separate:

- page clusters: one canonical page or deduped hit family
- semantic metadata: document family, family kind, topic cluster, topical support, and alignment

That lets the agent answer questions like:

- Is this page official?
- Is it actually about the requested topic?
- Is it recent enough?
- Is this one-off noise or part of a repeated topic thread?
- Do we still need another DDGS pass or a native search supplement?

## Design posture

- `ddgs` first, native search second
- deterministic structure before free-form synthesis
- visible gaps instead of bluffing
- better to leave `fresh_update` empty than let off-topic recency noise occupy it

## License

[MIT](./LICENSE)
