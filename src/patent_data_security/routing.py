"""Deterministic DOCS keyword and IPC routing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import ahocorasick

from patent_data_security.records import PatentRecord
from patent_data_security.taxonomy import TaxonomyBundle

TIER_RANK = {"E": 0, "R": 1, "W": 2, "S": 3}
IPC_PATTERN = re.compile(r"([A-HY]\d{2}[A-Z])\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ContextHit:
    context_id: str
    kind: str
    keyword: str
    start: int
    end: int
    distance: int
    snippet: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class KeywordHit:
    group_id: str
    tier: str
    keyword: str
    field: str
    snippet: str
    source_ids: tuple[str, ...]
    context_ids: tuple[str, ...] = ()
    context_scope: str = ""
    context_snippet: str = ""
    context_hits: tuple[ContextHit, ...] = ()


@dataclass(frozen=True)
class IPCHit:
    rule_id: str
    tier: str
    original_symbol: str
    normalized_symbol: str
    rule_symbol: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticHit:
    pattern_id: str
    keyword: str
    field: str
    snippet: str


@dataclass(frozen=True)
class RoutingResult:
    keyword_level: str
    ipc_level: str
    route_level: str
    keyword_hits: tuple[KeywordHit, ...]
    ipc_hits: tuple[IPCHit, ...]
    diagnostic_hits: tuple[DiagnosticHit, ...]
    normalized_ipc: tuple[str, ...]

    def keyword_hits_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.keyword_hits]

    def ipc_hits_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.ipc_hits]

    def diagnostics_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.diagnostic_hits]


@dataclass(frozen=True)
class KeywordRoutingResult:
    keyword_level: str
    keyword_hits: tuple[KeywordHit, ...]
    diagnostic_hits: tuple[DiagnosticHit, ...]

    def keyword_hits_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.keyword_hits]

    def diagnostics_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(hit) for hit in self.diagnostic_hits]


class PatentRouter:
    """Compile taxonomy rules once and route many patent records."""

    def __init__(self, taxonomy: TaxonomyBundle) -> None:
        self.taxonomy = taxonomy
        self.context_window = int(
            taxonomy.docs.get("normalization", {}).get("context_window_chars", 48)
        )
        self._groups_by_keyword: dict[str, dict[str, Any]] = {}
        for group in taxonomy.docs["term_groups"]:
            if group["tier"] == "E":
                continue
            for keyword in group["keywords"]:
                self._groups_by_keyword[normalize_text(keyword)] = group
        self._term_automaton = _build_automaton(self._groups_by_keyword)

        self._context_lexicons = {
            lexicon["id"]: lexicon for lexicon in taxonomy.docs["context_lexicons"]
        }
        self._context_patterns = {
            lexicon["id"]: _compile_keywords(
                {normalize_text(keyword): lexicon for keyword in lexicon["keywords"]}
            )
            for lexicon in taxonomy.docs["context_lexicons"]
        }
        self._diagnostics_by_keyword: dict[str, dict[str, Any]] = {}
        for pattern in taxonomy.docs.get("diagnostic_patterns", []):
            for keyword in pattern["keywords"]:
                self._diagnostics_by_keyword[normalize_text(keyword)] = pattern
        self._diagnostic_automaton = _build_automaton(self._diagnostics_by_keyword)
        self._ipc_rules: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for rule in taxonomy.ipc["rules"]:
            prepared = _prepare_ipc_rule(rule)
            self._ipc_rules.setdefault(prepared["parts"][:2], []).append(prepared)

    def route(self, record: PatentRecord) -> RoutingResult:
        keyword_routing = self.route_keywords(record)
        ipc_hits, normalized_ipc = self.route_ipc(record)
        ipc_level = highest_tier(hit.tier for hit in ipc_hits)
        route_level = highest_tier((keyword_routing.keyword_level, ipc_level))
        return RoutingResult(
            keyword_level=keyword_routing.keyword_level,
            ipc_level=ipc_level,
            route_level=route_level,
            keyword_hits=keyword_routing.keyword_hits,
            ipc_hits=tuple(ipc_hits),
            diagnostic_hits=keyword_routing.diagnostic_hits,
            normalized_ipc=tuple(normalized_ipc),
        )

    def route_keywords(self, record: PatentRecord) -> KeywordRoutingResult:
        """Extract DOCS keyword evidence and its local context associations."""

        keyword_hits: list[KeywordHit] = []
        diagnostics: list[DiagnosticHit] = []
        for field in ("abstract", "claim"):
            text = normalize_text(record.get(field))
            if not text:
                continue
            keyword_hits.extend(self._match_terms(text, field))
            diagnostics.extend(self._match_diagnostics(text, field))

        keyword_level = highest_tier(hit.tier for hit in keyword_hits)
        return KeywordRoutingResult(
            keyword_level=keyword_level,
            keyword_hits=tuple(keyword_hits),
            diagnostic_hits=tuple(diagnostics),
        )

    def route_ipc(self, record: PatentRecord) -> tuple[list[IPCHit], list[str]]:
        """Extract normalized IPC symbols and their most-specific taxonomy rules."""

        return self._match_ipc(record.get("ipc"), record.get("main_ipc"))

    def _match_terms(self, text: str, field: str) -> list[KeywordHit]:
        if self._term_automaton is None:
            return []
        hits: list[KeywordHit] = []
        seen: set[tuple[str, str, str]] = set()
        for start, end, keyword in _automaton_matches(self._term_automaton, text):
            group = self._groups_by_keyword[keyword]
            tier = group["tier"]
            policy = group["match_policy"]
            identity = (group["id"], keyword, field)
            if identity in seen:
                continue
            scope_start, scope_end, scope_mode = _context_scope(
                text, start, end, self.context_window
            )
            context_hits = self._match_contexts(
                text,
                keyword_start=start,
                keyword_end=end,
                scope_start=scope_start,
                scope_end=scope_end,
            )
            matched_context_ids = {hit.context_id for hit in context_hits}
            context_ids: list[str] = []
            if policy["mode"] == "cooccurrence":
                context_ids = [
                    context_id
                    for context_id in policy.get("required_any", [])
                    if context_id in matched_context_ids
                ]
                if not context_ids:
                    continue
            seen.add(identity)
            hits.append(
                KeywordHit(
                    group_id=group["id"],
                    tier=tier,
                    keyword=text[start:end],
                    field=field,
                    snippet=_snippet(text, start, end),
                    source_ids=tuple(group["source_ids"]),
                    context_ids=tuple(context_ids),
                    context_scope=scope_mode,
                    context_snippet=_bounded_scope_snippet(
                        text, scope_start, scope_end, start, end
                    ),
                    context_hits=tuple(context_hits),
                )
            )
        return hits

    def _match_contexts(
        self,
        text: str,
        *,
        keyword_start: int,
        keyword_end: int,
        scope_start: int,
        scope_end: int,
    ) -> list[ContextHit]:
        scope = text[scope_start:scope_end]
        hits: list[ContextHit] = []
        for context_id, pattern in self._context_patterns.items():
            if pattern is None:
                continue
            lexicon = self._context_lexicons[context_id]
            for match in pattern.finditer(scope):
                start = scope_start + match.start()
                end = scope_start + match.end()
                if _spans_overlap(start, end, keyword_start, keyword_end):
                    continue
                hits.append(
                    ContextHit(
                        context_id=context_id,
                        kind=lexicon["kind"],
                        keyword=text[start:end],
                        start=start,
                        end=end,
                        distance=_span_distance(start, end, keyword_start, keyword_end),
                        snippet=_snippet(text, start, end),
                        source_ids=tuple(lexicon["source_ids"]),
                    )
                )
        hits.sort(key=lambda hit: (hit.start, -(hit.end - hit.start), hit.context_id))
        return hits

    def _match_diagnostics(self, text: str, field: str) -> list[DiagnosticHit]:
        if self._diagnostic_automaton is None:
            return []
        hits: list[DiagnosticHit] = []
        seen: set[tuple[str, str, str]] = set()
        for start, end, keyword in _automaton_matches(self._diagnostic_automaton, text):
            pattern = self._diagnostics_by_keyword[keyword]
            identity = (pattern["id"], keyword, field)
            if identity in seen:
                continue
            seen.add(identity)
            hits.append(
                DiagnosticHit(
                    pattern_id=pattern["id"],
                    keyword=text[start:end],
                    field=field,
                    snippet=_snippet(text, start, end),
                )
            )
        return hits

    def _match_ipc(self, *raw_fields: str) -> tuple[list[IPCHit], list[str]]:
        symbols: dict[str, str] = {}
        for raw_field in raw_fields:
            for match in IPC_PATTERN.finditer(normalize_text(raw_field).upper()):
                normalized = f"{match.group(1).upper()} {int(match.group(2))}/{match.group(3)}"
                symbols.setdefault(normalized, match.group(0))

        hits: list[IPCHit] = []
        for symbol, original in symbols.items():
            code = _split_ipc_symbol(symbol)
            matching = [
                rule
                for rule in self._ipc_rules.get(code[:2], [])
                if _ipc_rule_matches(rule, code)
            ]
            if not matching:
                continue
            best = max(matching, key=lambda rule: rule["specificity"])
            hits.append(
                IPCHit(
                    rule_id=best["id"],
                    tier=best["tier"],
                    original_symbol=original,
                    normalized_symbol=symbol,
                    rule_symbol=best["symbol"],
                    source_ids=tuple(best["source_ids"]),
                )
            )
        return hits, list(symbols)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = re.sub(r"[\t\u3000 ]+", " ", normalized)
    return normalized.strip()


def highest_tier(tiers: Any) -> str:
    return max(tiers, key=TIER_RANK.get, default="E")


def _compile_keywords(mapping: dict[str, Any]) -> re.Pattern[str] | None:
    if not mapping:
        return None
    alternatives = sorted(mapping, key=lambda keyword: (-len(keyword), keyword))
    return re.compile("|".join(re.escape(keyword) for keyword in alternatives), re.IGNORECASE)


def _build_automaton(mapping: dict[str, Any]) -> ahocorasick.Automaton | None:
    if not mapping:
        return None
    automaton = ahocorasick.Automaton()
    for keyword in mapping:
        automaton.add_word(keyword, keyword)
    automaton.make_automaton()
    return automaton


def _automaton_matches(
    automaton: ahocorasick.Automaton, text: str
) -> list[tuple[int, int, str]]:
    matches = [
        (end - len(keyword) + 1, end + 1, keyword)
        for end, keyword in automaton.iter(text)
    ]
    matches.sort(key=lambda match: (match[0], -(match[1] - match[0]), match[2]))
    selected: list[tuple[int, int, str]] = []
    next_start = 0
    for match in matches:
        if match[0] < next_start:
            continue
        selected.append(match)
        next_start = match[1]
    return selected


def _context_scope(text: str, start: int, end: int, size: int) -> tuple[int, int, str]:
    """Prefer the complete sentence; fall back to a fixed window without punctuation."""

    boundaries = "。！？!?；;\n\r"
    sentence_left = max(text.rfind(mark, 0, start) for mark in boundaries)
    sentence_ends = [
        position
        for mark in boundaries
        if (position := text.find(mark, end)) >= 0
    ]
    if sentence_left >= 0 or sentence_ends:
        left = sentence_left + 1 if sentence_left >= 0 else 0
        right = min(sentence_ends) if sentence_ends else len(text)
        return left, right, "sentence"
    return max(0, start - size), min(len(text), end + size), "window"


def _bounded_scope_snippet(
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


def _spans_overlap(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def _span_distance(start: int, end: int, other_start: int, other_end: int) -> int:
    if end <= other_start:
        return other_start - end
    if start >= other_end:
        return start - other_end
    return 0


def _snippet(text: str, start: int, end: int, size: int = 36) -> str:
    return text[max(0, start - size) : min(len(text), end + size)].replace("\n", " ")


def _split_ipc_symbol(symbol: str) -> tuple[str, int, str]:
    match = IPC_PATTERN.fullmatch(symbol)
    if match is None:
        raise ValueError(f"Invalid IPC symbol: {symbol}")
    return match.group(1).upper(), int(match.group(2)), match.group(3)


def _prepare_ipc_rule(rule: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(rule)
    subclass, main_group, subgroup = _split_ipc_symbol(rule["symbol"])
    prepared["parts"] = (subclass, main_group, subgroup)
    prepared["specificity"] = (
        len(subgroup.rstrip("0")),
        1 if rule["match"] == "exact" else 0,
    )
    return prepared


def _ipc_rule_matches(rule: dict[str, Any], code: tuple[str, int, str]) -> bool:
    if rule["parts"][:2] != code[:2]:
        return False
    rule_subgroup = rule["parts"][2]
    code_subgroup = code[2]
    if rule["match"] == "exact":
        return rule_subgroup == code_subgroup
    if set(rule_subgroup) == {"0"}:
        return True
    return code_subgroup.startswith(rule_subgroup)
