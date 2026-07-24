# Step 1 public reference implementation

This folder is a shareable implementation of the paper's first-stage routing procedure. It
keeps the reproducible method while excluding assets that are not part of the public release.

## What is included

1. Unicode NFKC normalization, Latin case folding, whitespace normalization, and connector
   normalization.
2. Longest-first phrase matching over `claim`, `abstract`, and `title`.
3. Standalone and local-context co-occurrence rules.
4. Explicit exclusion phrases and diagnostic-only patterns.
5. IPC audit rules that never change the S/E route.
6. One-record-per-patent deduplication, with an S record preferred over an E duplicate.
7. Deterministic SHA-256 sampling: all S records and a configured probability sample of E.
8. A result CSV and a manifest containing counts, sampling parameters, and file hashes.

The implementation makes no LLM or network request.

## What is intentionally excluded

- The production keyword taxonomy and its variants.
- The expert lexicon and unpublished validation examples.
- Raw patent data and its storage address.
- Private source manifests, credentials, API keys, and intermediate project results.

The distributed `rules.template.json` therefore contains empty rule lists. This prevents the
public repository from silently presenting a placeholder taxonomy as the paper's validated
production taxonomy.

## Input

`config.example.json` contains:

```json
{
  "input_csv": ""
}
```

The original data source address is intentionally blank. Copy the file to a local, untracked
configuration and fill in your own CSV path and column mapping, or pass the input on the command
line. The canonical fields are:

```text
patent_id,title,abstract,claim,ipc,applicant,application_date
```

## Rules

Populate a copy of `rules.template.json` with terms that are cleared for public release. A
standalone concept has this shape:

```json
{
  "concept_id": "PUBLIC-CONCEPT-001",
  "category": "technical",
  "canonical_term": "public descriptive name",
  "variants": ["public phrase"],
  "match_policy": {"mode": "standalone"},
  "excluded_phrases": [],
  "public_source_ids": ["public-citation-id"]
}
```

A context-dependent concept uses:

```json
{
  "concept_id": "PUBLIC-CONCEPT-002",
  "category": "descriptive",
  "canonical_term": "public descriptive name",
  "variants": ["ambiguous phrase"],
  "match_policy": {
    "mode": "cooccurrence",
    "required_any": ["PUBLIC-CONTEXT-001"]
  }
}
```

The referenced context must appear in the same sentence, or inside the configured character
window when the text has no sentence boundary.

## Run

From the repository root:

```bash
python -m loop.step1.step1_public \
  --config loop/step1/config.example.json \
  --input /path/to/your/patents.csv \
  --output-dir loop/step1/output
```

`result.csv` contains the S/E route, explicit Step 1 label, inclusion probability, inverse
probability weight, and JSON audit fields. `manifest.json` deliberately records
`"input_source": ""`; it stores only the input file name and SHA-256, avoiding disclosure of a
local or licensed-data address.

## Reproducibility note

For an E record with identity `dataset_id|patent_id`, inclusion is determined by:

```text
u = uint64(SHA256(seed + "|" + identity)[0:8]) / 2^64
selected = u < e_sample_rate
```

The same input, rules, seed, and sampling rate therefore produce the same selected patent IDs.
