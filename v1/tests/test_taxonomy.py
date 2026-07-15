import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
TAXONOMY_DIR = ROOT / "config" / "taxonomy"
VALID_TIERS = {"S", "W", "R", "E"}


def load_json(name: str) -> dict:
    with (TAXONOMY_DIR / name).open(encoding="utf-8") as file:
        return json.load(file)


def source_ids() -> set[str]:
    return {source["id"] for source in load_json("sources.json")["sources"]}


def assert_known_sources(ids: list[str], known: set[str]) -> None:
    assert ids
    assert set(ids) <= known


def test_source_registry_is_unique_and_auditable() -> None:
    registry = load_json("sources.json")
    sources = registry["sources"]
    ids = [source["id"] for source in sources]

    assert len(ids) == len(set(ids))
    for source in sources:
        assert source["title"].strip()
        assert source["publisher"].strip()
        assert source["locators"]
        assert source["contribution"].strip()
        assert source.get("url") or source.get("local_reference")


def test_docs_terms_have_one_tier_and_valid_sources() -> None:
    taxonomy = load_json("docs_taxonomy.json")
    known = source_ids()
    seen_keywords: dict[str, str] = {}

    assert set(taxonomy["tier_definitions"]) == VALID_TIERS
    assert taxonomy["docs_fields"] == ["abstract", "claim"]

    for lexicon in taxonomy["context_lexicons"]:
        assert lexicon["keywords"]
        assert_known_sources(lexicon["source_ids"], known)

    for group in taxonomy["term_groups"]:
        assert group["tier"] in VALID_TIERS
        assert group["keywords"]
        assert group["match_policy"]["mode"]
        assert_known_sources(group["source_ids"], known)
        for keyword in group["keywords"]:
            assert keyword == keyword.strip()
            assert keyword not in seen_keywords, (
                f"keyword {keyword!r} appears in both {seen_keywords[keyword]} and {group['id']}"
            )
            seen_keywords[keyword] = group["id"]


def test_high_risk_docs_terms_have_conservative_tiers() -> None:
    taxonomy = load_json("docs_taxonomy.json")
    tier_by_keyword = {
        keyword: group["tier"]
        for group in taxonomy["term_groups"]
        for keyword in group["keywords"]
    }

    assert tier_by_keyword["数据安全"] == "S"
    assert tier_by_keyword["差分隐私"] == "S"
    assert tier_by_keyword["安全多方计算"] == "S"
    assert tier_by_keyword["联邦学习"] == "S"
    assert tier_by_keyword["个人信息保护"] == "W"
    assert tier_by_keyword["加密"] == "R"
    assert tier_by_keyword["区块链"] == "R"
    assert tier_by_keyword["数据"] == "E"
    assert "安全生产" not in tier_by_keyword

    diagnostic_keywords = {
        keyword
        for pattern in taxonomy["diagnostic_patterns"]
        for keyword in pattern["keywords"]
    }
    assert "安全生产" in diagnostic_keywords


def test_ipc_rules_are_versioned_and_sourced() -> None:
    taxonomy = load_json("ipc_taxonomy.json")
    known = source_ids()
    rule_ids: set[str] = set()
    symbol_pattern = re.compile(r"^[A-H]\d{2}[A-Z] \d+/\d+$")

    assert taxonomy["default_tier"] == "E"
    assert set(taxonomy["tier_definitions"]) == VALID_TIERS
    assert taxonomy["supported_editions"] == ["2021.01", "2026.01"]

    for rule in taxonomy["rules"]:
        assert rule["id"] not in rule_ids
        rule_ids.add(rule["id"])
        assert rule["tier"] in VALID_TIERS
        assert rule["match"] in {"exact", "subtree"}
        assert symbol_pattern.fullmatch(rule["symbol"])
        assert set(rule["editions"]) <= set(taxonomy["supported_editions"])
        assert rule["official_title"].strip()
        assert_known_sources(rule["source_ids"], known)

    assert_known_sources(taxonomy["default_rule"]["source_ids"], known)


def test_core_ipc_rules_have_expected_tiers() -> None:
    taxonomy = load_json("ipc_taxonomy.json")
    tier_by_symbol = {}
    for rule in taxonomy["rules"]:
        tier_by_symbol.setdefault(rule["symbol"], set()).add(rule["tier"])

    assert tier_by_symbol["G06F 21/60"] == {"S"}
    assert tier_by_symbol["G06F 21/62"] == {"S"}
    assert tier_by_symbol["G06F 21/64"] == {"S"}
    assert "W" in tier_by_symbol["H04L 9/00"]
    assert "G06F 16/00" not in tier_by_symbol
    assert "W" in tier_by_symbol["G06F 11/14"]
    assert "R" in tier_by_symbol["G06F 21/10"]
