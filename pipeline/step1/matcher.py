"""Deterministic S/E keyword matching with auditable local context."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from typing import Any

import ahocorasick

from pipeline.common.records import PatentRecord
from pipeline.step1.taxonomy import KeywordBundle

IPC_PATTERN = re.compile(r"([A-HY]\d{2}[A-Z])\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
ASCII_WORD = re.compile(r"^[a-z0-9][a-z0-9 ._+/-]*$", re.IGNORECASE)
SENTENCE_BOUNDARIES = "。！？!?；;\n\r"
CONNECTOR_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)


@dataclass(frozen=True)
class ContextHit:
    context_id: str
    kind: str
    matched_text: str
    start: int
    end: int
    distance: int
    occurrence_count: int
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class KeywordHit:
    concept_id: str
    keyword_id: str
    category: str
    canonical_term: str
    matched_text: str
    field: str
    start: int
    end: int
    occurrence_count: int
    match_policy: str
    context_scope: str
    context_snippet: str
    context_hits: tuple[ContextHit, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticHit:
    pattern_id: str
    matched_text: str
    field: str
    start: int
    end: int
    snippet: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class IPCAuditHit:
    rule_id: str
    original_symbol: str
    normalized_symbol: str
    rule_symbol: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class MatchResult:
    route: str
    keyword_hits: tuple[KeywordHit, ...]
    diagnostic_hits: tuple[DiagnosticHit, ...]
    ipc_audit_hits: tuple[IPCAuditHit, ...]

    @property
    def valid_hit_count(self) -> int:
        return len(self.keyword_hits)

    @property
    def descriptive_hit_count(self) -> int:
        return sum(hit.category == "descriptive" for hit in self.keyword_hits)

    @property
    def technical_hit_count(self) -> int:
        return sum(hit.category == "technical" for hit in self.keyword_hits)

    @property
    def matched_concepts(self) -> tuple[str, ...]:
        return tuple(sorted({hit.concept_id for hit in self.keyword_hits}))

    def keyword_hits_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.keyword_hits]

    def context_hits_jsonable(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for hit in self.keyword_hits:
            for context in hit.context_hits:
                values.append(
                    {
                        "concept_id": hit.concept_id,
                        "keyword_id": hit.keyword_id,
                        "field": hit.field,
                        **asdict(context),
                    }
                )
        return values

    def diagnostics_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.diagnostic_hits]

    def ipc_audit_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.ipc_audit_hits]


class KeywordMatcher:
    """Compile versioned rules once, then match many patent records."""

    def __init__(self, bundle: KeywordBundle) -> None:
        self.bundle = bundle
        config = bundle.keywords
        self.fields = tuple(config["matching"]["fields"])
        self.context_window = int(config["matching"]["context_window_chars"])

        self._concepts_by_keyword: dict[str, dict[str, Any]] = {}
        self._keyword_ids: dict[tuple[str, str], str] = {}
        self._excluded_phrases: dict[str, tuple[str, ...]] = {}
        for concept in config["concepts"]:
            concept_id = concept["concept_id"]
            self._excluded_phrases[concept_id] = tuple(
                normalize_text(value) for value in concept.get("excluded_phrases", [])
            )
            for index, variant in enumerate(concept["variants"], start=1):
                keyword = normalize_text(variant)
                self._concepts_by_keyword[keyword] = concept
                self._keyword_ids[(concept_id, keyword)] = f"{concept_id}:K{index:03d}"
        self._term_automaton = _build_automaton(self._concepts_by_keyword)

        self._contexts_by_keyword: dict[str, list[dict[str, Any]]] = {}
        for lexicon in config.get("context_lexicons", []):
            for variant in lexicon["variants"]:
                self._contexts_by_keyword.setdefault(normalize_text(variant), []).append(lexicon)
        self._context_automaton = _build_automaton(self._contexts_by_keyword)

        self._diagnostics_by_keyword: dict[str, dict[str, Any]] = {}
        for pattern in config.get("diagnostic_patterns", []):
            for variant in pattern["variants"]:
                self._diagnostics_by_keyword[normalize_text(variant)] = pattern
        self._diagnostic_automaton = _build_automaton(self._diagnostics_by_keyword)

        self._ipc_rules = [_prepare_ipc_rule(rule) for rule in config.get("ipc_audit_rules", [])]

    def match(self, record: PatentRecord) -> MatchResult:
        """Return S when any validated keyword/context hit exists; otherwise E."""

        hits: list[KeywordHit] = []
        diagnostics: list[DiagnosticHit] = []
        for field in self.fields:
            text = normalize_text(record.get(field))
            if not text:
                continue
            hits.extend(self._match_field(text, field))
            diagnostics.extend(self._match_diagnostics(text, field))
        ipc_hits = self._match_ipc(record.get("ipc"), record.get("main_ipc"))
        hits.sort(key=lambda hit: (self.fields.index(hit.field), hit.start, hit.concept_id))
        diagnostics.sort(
            key=lambda hit: (self.fields.index(hit.field), hit.start, hit.pattern_id)
        )
        return MatchResult(
            route="S" if hits else "E",
            keyword_hits=tuple(hits),
            diagnostic_hits=tuple(diagnostics),
            ipc_audit_hits=tuple(ipc_hits),
        )

    def _match_field(self, text: str, field: str) -> list[KeywordHit]:
        aggregated: dict[tuple[str, str, str], KeywordHit] = {}
        for start, end, keyword in _nonoverlapping_matches(self._term_automaton, text):
            concept = self._concepts_by_keyword[keyword]
            if self._is_excluded(text, start, end, concept["concept_id"]):
                continue
            scope_start, scope_end, scope = _context_scope(
                text,
                start,
                end,
                field,
                self.context_window,
            )
            context_hits = self._match_contexts(text, start, end, scope_start, scope_end)
            matched_context_ids = {hit.context_id for hit in context_hits}
            policy = concept["match_policy"]
            if policy["mode"] == "cooccurrence":
                required_any = set(policy.get("required_any", []))
                required_all = set(policy.get("required_all", []))
                if required_any and not (required_any & matched_context_ids):
                    continue
                if required_all and not required_all.issubset(matched_context_ids):
                    continue
            concept_id = concept["concept_id"]
            keyword_id = self._keyword_ids[(concept_id, keyword)]
            identity = (concept_id, keyword_id, field)
            existing = aggregated.get(identity)
            if existing is not None:
                aggregated[identity] = replace(
                    existing,
                    occurrence_count=existing.occurrence_count + 1,
                )
                continue
            aggregated[identity] = KeywordHit(
                concept_id=concept_id,
                keyword_id=keyword_id,
                category=concept["category"],
                canonical_term=concept["canonical_term"],
                matched_text=text[start:end],
                field=field,
                start=start,
                end=end,
                occurrence_count=1,
                match_policy=policy["mode"],
                context_scope=scope,
                context_snippet=_bounded_snippet(text, scope_start, scope_end, start, end),
                context_hits=tuple(context_hits),
                source_ids=tuple(concept["source_ids"]),
            )
        return list(aggregated.values())

    def _match_contexts(
        self,
        text: str,
        keyword_start: int,
        keyword_end: int,
        scope_start: int,
        scope_end: int,
    ) -> list[ContextHit]:
        scope = text[scope_start:scope_end]
        aggregated: dict[tuple[str, str], ContextHit] = {}
        for relative_start, relative_end, variant in _nonoverlapping_matches(
            self._context_automaton, scope
        ):
            start = scope_start + relative_start
            end = scope_start + relative_end
            if _spans_overlap(start, end, keyword_start, keyword_end):
                continue
            for lexicon in self._contexts_by_keyword[variant]:
                identity = (lexicon["id"], variant)
                candidate = ContextHit(
                    context_id=lexicon["id"],
                    kind=lexicon["kind"],
                    matched_text=text[start:end],
                    start=start,
                    end=end,
                    distance=_span_distance(start, end, keyword_start, keyword_end),
                    occurrence_count=1,
                    source_ids=tuple(lexicon["source_ids"]),
                )
                existing = aggregated.get(identity)
                if existing is None:
                    aggregated[identity] = candidate
                elif candidate.distance < existing.distance:
                    aggregated[identity] = replace(
                        candidate,
                        occurrence_count=existing.occurrence_count + 1,
                    )
                else:
                    aggregated[identity] = replace(
                        existing,
                        occurrence_count=existing.occurrence_count + 1,
                    )
        hits = list(aggregated.values())
        hits.sort(key=lambda hit: (hit.start, -(hit.end - hit.start), hit.context_id))
        return hits

    def _match_diagnostics(self, text: str, field: str) -> list[DiagnosticHit]:
        hits: list[DiagnosticHit] = []
        seen: set[tuple[str, str]] = set()
        for start, end, keyword in _nonoverlapping_matches(self._diagnostic_automaton, text):
            pattern = self._diagnostics_by_keyword[keyword]
            identity = (pattern["id"], keyword)
            if identity in seen:
                continue
            seen.add(identity)
            hits.append(
                DiagnosticHit(
                    pattern_id=pattern["id"],
                    matched_text=text[start:end],
                    field=field,
                    start=start,
                    end=end,
                    snippet=_snippet(text, start, end),
                    source_ids=tuple(pattern["source_ids"]),
                )
            )
        return hits

    def _match_ipc(self, *values: str) -> list[IPCAuditHit]:
        symbols: dict[str, str] = {}
        for value in values:
            for match in IPC_PATTERN.finditer(normalize_text(value).upper()):
                normalized = f"{match.group(1).upper()} {int(match.group(2))}/{match.group(3)}"
                symbols.setdefault(normalized, match.group(0))
        hits: list[IPCAuditHit] = []
        for normalized, original in symbols.items():
            code = _split_ipc_symbol(normalized)
            matches = [rule for rule in self._ipc_rules if _ipc_rule_matches(rule, code)]
            if not matches:
                continue
            rule = max(matches, key=lambda item: item["specificity"])
            hits.append(
                IPCAuditHit(
                    rule_id=rule["id"],
                    original_symbol=original,
                    normalized_symbol=normalized,
                    rule_symbol=rule["symbol"],
                    source_ids=tuple(rule["source_ids"]),
                )
            )
        hits.sort(key=lambda hit: hit.normalized_symbol)
        return hits

    def _is_excluded(self, text: str, start: int, end: int, concept_id: str) -> bool:
        for phrase in self._excluded_phrases[concept_id]:
            search_start = max(0, start - len(phrase) + 1)
            position = text.find(phrase, search_start, min(len(text), end + len(phrase)))
            while position >= 0:
                phrase_end = position + len(phrase)
                if position <= start and phrase_end >= end:
                    return True
                position = text.find(
                    phrase,
                    position + 1,
                    min(len(text), end + len(phrase)),
                )
        return False


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = normalized.translate(CONNECTOR_TRANSLATION)
    normalized = re.sub(r"[\t\u3000 ]+", " ", normalized)
    return normalized.strip()


def _build_automaton(mapping: dict[str, Any]) -> ahocorasick.Automaton | None:
    if not mapping:
        return None
    automaton = ahocorasick.Automaton()
    for keyword in mapping:
        automaton.add_word(keyword, keyword)
    automaton.make_automaton()
    return automaton


def _nonoverlapping_matches(
    automaton: ahocorasick.Automaton | None,
    text: str,
) -> list[tuple[int, int, str]]:
    if automaton is None:
        return []
    matches = []
    for end_index, keyword in automaton.iter(text):
        start = end_index - len(keyword) + 1
        end = end_index + 1
        if _needs_ascii_boundary(keyword) and not _has_ascii_boundaries(text, start, end):
            continue
        matches.append((start, end, keyword))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    selected: list[tuple[int, int, str]] = []
    cursor = 0
    for match in matches:
        if match[0] < cursor:
            continue
        selected.append(match)
        cursor = match[1]
    return selected


def _needs_ascii_boundary(keyword: str) -> bool:
    return bool(ASCII_WORD.fullmatch(keyword)) and any(char.isascii() for char in keyword)


def _has_ascii_boundaries(text: str, start: int, end: int) -> bool:
    left_valid = start == 0 or not text[start - 1].isascii() or not text[start - 1].isalnum()
    right_valid = end == len(text) or not text[end].isascii() or not text[end].isalnum()
    return left_valid and right_valid


def _context_scope(
    text: str,
    start: int,
    end: int,
    field: str,
    size: int,
) -> tuple[int, int, str]:
    if field == "title":
        return 0, len(text), "title"
    left_boundary = max(text.rfind(mark, 0, start) for mark in SENTENCE_BOUNDARIES)
    right_boundaries = [
        position
        for mark in SENTENCE_BOUNDARIES
        if (position := text.find(mark, end)) >= 0
    ]
    if left_boundary >= 0 or right_boundaries:
        left = left_boundary + 1 if left_boundary >= 0 else 0
        right = min(right_boundaries) if right_boundaries else len(text)
        return left, right, "sentence"
    return max(0, start - size), min(len(text), end + size), "window"


def _bounded_snippet(
    text: str,
    scope_start: int,
    scope_end: int,
    keyword_start: int,
    keyword_end: int,
    size: int = 120,
) -> str:
    left = max(scope_start, keyword_start - size)
    right = min(scope_end, keyword_end + size)
    prefix = "…" if left > scope_start else ""
    suffix = "…" if right < scope_end else ""
    return prefix + text[left:right].replace("\n", " ") + suffix


def _snippet(text: str, start: int, end: int, size: int = 36) -> str:
    return text[max(0, start - size) : min(len(text), end + size)].replace("\n", " ")


def _spans_overlap(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def _span_distance(start: int, end: int, other_start: int, other_end: int) -> int:
    if end <= other_start:
        return other_start - end
    if start >= other_end:
        return start - other_end
    return 0


def _split_ipc_symbol(symbol: str) -> tuple[str, int, str]:
    match = IPC_PATTERN.fullmatch(symbol)
    if match is None:
        raise ValueError(f"Invalid IPC symbol: {symbol}")
    return match.group(1).upper(), int(match.group(2)), match.group(3)


def _prepare_ipc_rule(rule: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(rule)
    prepared["parts"] = _split_ipc_symbol(rule["symbol"])
    rank = {"subclass": 1, "main_group": 2, "exact": 3}[rule["match"]]
    prepared["specificity"] = (rank, len(prepared["parts"][2].rstrip("0")))
    return prepared


def _ipc_rule_matches(rule: dict[str, Any], code: tuple[str, int, str]) -> bool:
    mode = rule["match"]
    if mode == "subclass":
        return rule["parts"][0] == code[0]
    if mode == "main_group":
        return rule["parts"][:2] == code[:2]
    if mode == "exact":
        return rule["parts"] == code
    raise ValueError(f"Unsupported IPC match mode: {mode}")
