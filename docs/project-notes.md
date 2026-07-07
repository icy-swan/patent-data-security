# Project Notes

## Scope

This project classifies patent records from CSV files and determines whether each record belongs to the data security domain.

## Initial Design Notes

- Keep raw data outside Git.
- Keep prompts versioned under `prompts/`.
- Keep model output structured so it can be audited later.
- Prefer deterministic LLM settings for repeatable classification.

## Open Questions

- Which patent fields are always available in the CSV?
- Should the classifier support multiple LLM providers?
- What confidence threshold should require manual review?

