# Codex 2021 review tool

`codex_review_2021.py` deterministically reproduces the completed Codex review of the frozen
5,000-row 2021 Step-3 sample. It is dataset-specific: the 99 reviewed label changes are keyed by
the opaque `sample_id`; `patent_id` is never used to infer technical content.

Default input:

```text
data/step3/need_manual_review_positive.csv
```

Default output:

```text
data/step3/codex_result_positive.csv
```

Run from the repository root:

```bash
python tools/codex_review_2021.py
```

Rerun and atomically replace an existing output:

```bash
python tools/codex_review_2021.py --force
```

Validate an existing result without changing it:

```bash
python tools/codex_review_2021.py --validate-only
```

The output preserves every source field and appends exactly:

- `codex_review_label`: explicit reviewed label, restricted to `DATA_SECURITY` or `OTHER`.
- `codex_review_reason`: reviewed label, an exact title/abstract/claim excerpt, and the reason.

Validation checks the row and column structure, verifies all source fields remain byte-equivalent
after CSV parsing, verifies every quoted excerpt exists in its named source field, checks label/reason
consistency, and rejects any reason containing `patent_id`.
