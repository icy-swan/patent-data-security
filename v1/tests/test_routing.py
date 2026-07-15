from patent_data_security.records import PatentRecord
from patent_data_security.routing import PatentRouter
from patent_data_security.taxonomy import load_taxonomies


def record(**values: str) -> PatentRecord:
    return PatentRecord(row_number=2, values=values, raw={})


def test_longest_keyword_prevents_nested_generic_crypto_hit() -> None:
    result = PatentRouter(load_taxonomies()).route(
        record(abstract="采用同态加密完成联合统计", claim="", ipc="", main_ipc="")
    )

    assert result.keyword_level == "S"
    assert {(hit.keyword, hit.tier) for hit in result.keyword_hits} == {("同态加密", "S")}


def test_cooccurrence_term_requires_local_data_context() -> None:
    router = PatentRouter(load_taxonomies())

    without_context = router.route(
        record(abstract="一种零知识证明装置", claim="", ipc="", main_ipc="")
    )
    with_context = router.route(
        record(abstract="一种保护用户数据的零知识证明方法", claim="", ipc="", main_ipc="")
    )

    assert without_context.keyword_level == "E"
    assert with_context.keyword_level == "W"
    assert with_context.keyword_hits[0].context_ids == ("CTX-DATA-OBJECT",)


def test_context_retrieval_prefers_full_sentence_over_48_char_window() -> None:
    text = "用户数据" + "甲" * 60 + "采用零知识证明。下一句无关。"
    result = PatentRouter(load_taxonomies()).route(
        record(abstract=text, claim="", ipc="", main_ipc="")
    )

    hit = result.keyword_hits[0]
    assert result.keyword_level == "W"
    assert hit.context_scope == "sentence"
    assert hit.context_ids == ("CTX-DATA-OBJECT",)
    assert any(context.keyword == "用户数据" for context in hit.context_hits)


def test_context_retrieval_uses_48_char_window_without_sentence_punctuation() -> None:
    router = PatentRouter(load_taxonomies())
    within_window = router.route(
        record(
            abstract="用户数据" + "甲" * 40 + "零知识证明",
            claim="",
            ipc="",
            main_ipc="",
        )
    )
    outside_window = router.route(
        record(
            abstract="用户数据" + "甲" * 60 + "零知识证明",
            claim="",
            ipc="",
            main_ipc="",
        )
    )

    assert within_window.keyword_level == "W"
    assert within_window.keyword_hits[0].context_scope == "window"
    assert outside_window.keyword_level == "E"


def test_context_retrieval_does_not_cross_sentence_boundary() -> None:
    result = PatentRouter(load_taxonomies()).route(
        record(
            abstract="用户数据。采用零知识证明完成计算。",
            claim="",
            ipc="",
            main_ipc="",
        )
    )

    assert result.keyword_level == "E"


def test_standalone_keyword_retains_context_associations() -> None:
    result = PatentRouter(load_taxonomies()).route(
        record(abstract="用户数据采用加密保护。", claim="", ipc="", main_ipc="")
    )

    hit = result.keyword_hits[0]
    assert result.keyword_level == "R"
    assert hit.keyword == "加密"
    assert hit.context_ids == ()
    assert {(context.context_id, context.keyword) for context in hit.context_hits} >= {
        ("CTX-DATA-OBJECT", "用户数据"),
        ("CTX-PROTECTION-ACTION", "保护"),
    }


def test_neighbor_safety_diagnostic_never_lowers_route() -> None:
    result = PatentRouter(load_taxonomies()).route(
        record(
            abstract="本发明以数据安全方法保护食品安全追溯数据库",
            claim="",
            ipc="",
            main_ipc="",
        )
    )

    assert result.route_level == "S"
    assert result.diagnostic_hits
    assert result.diagnostic_hits[0].pattern_id == "DIAG-PHYSICAL-SAFETY"


def test_ipc_uses_most_specific_rule_and_ignores_generic_database_group() -> None:
    router = PatentRouter(load_taxonomies())

    specific = router.route(
        record(abstract="", claim="", ipc="G06F21/00; G06F21/62", main_ipc="G06F21/62")
    )
    generic_database = router.route(
        record(abstract="", claim="", ipc="G06F16/25", main_ipc="G06F16/25")
    )

    assert specific.ipc_level == "S"
    assert {hit.normalized_symbol for hit in specific.ipc_hits} == {
        "G06F 21/00",
        "G06F 21/62",
    }
    access_hit = next(
        hit for hit in specific.ipc_hits if hit.normalized_symbol.endswith("/62")
    )
    assert access_hit.tier == "S"
    assert generic_database.ipc_level == "E"


def test_routing_reads_abstract_and_claim_but_not_title() -> None:
    router = PatentRouter(load_taxonomies())
    title_only = router.route(
        record(title="数据安全方法", abstract="普通分析", claim="普通处理", ipc="", main_ipc="")
    )
    claim_hit = router.route(
        record(title="普通方法", abstract="", claim="采用联邦学习", ipc="", main_ipc="")
    )

    assert title_only.keyword_level == "E"
    assert claim_hit.keyword_level == "S"
    assert claim_hit.keyword_hits[0].field == "claim"
