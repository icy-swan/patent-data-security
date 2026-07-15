from pipeline.common.records import PatentRecord
from pipeline.step1.matcher import KeywordMatcher, normalize_text
from pipeline.step1.taxonomy import load_keyword_bundle


def record(**values: str) -> PatentRecord:
    return PatentRecord(row_number=2, values=values)


def matcher() -> KeywordMatcher:
    return KeywordMatcher(load_keyword_bundle())


def test_crypto_is_in_scope_and_longest_term_wins() -> None:
    result = matcher().match(
        record(claim="采用同态加密完成联合统计", abstract="", title="", ipc="", main_ipc="")
    )

    assert result.route == "S"
    assert [(hit.matched_text, hit.category) for hit in result.keyword_hits] == [
        ("同态加密", "technical")
    ]


def test_title_is_scanned_after_claim_and_abstract() -> None:
    result = matcher().match(
        record(claim="", abstract="", title="一种密码学协议", ipc="", main_ipc="")
    )

    assert result.route == "S"
    assert result.keyword_hits[0].field == "title"


def test_physical_safety_and_biological_codon_are_diagnostics_only() -> None:
    physical = matcher().match(
        record(
            claim="食品安全检测结果写入数据库",
            abstract="",
            title="",
            ipc="",
            main_ipc="",
        )
    )
    codon = matcher().match(
        record(claim="识别基因中的终止密码子", abstract="", title="", ipc="", main_ipc="")
    )

    assert physical.route == "E"
    assert physical.diagnostic_hits[0].pattern_id == "DIAG-PHYSICAL-SAFETY"
    assert codon.route == "E"
    assert codon.diagnostic_hits[0].pattern_id == "DIAG-BIOLOGICAL-CODON"


def test_generic_protection_requires_local_data_context() -> None:
    without_context = matcher().match(
        record(claim="采用保护装置避免齿轮损坏", abstract="", title="", ipc="", main_ipc="")
    )
    with_context = matcher().match(
        record(claim="采用隔离模块保护用户数据", abstract="", title="", ipc="", main_ipc="")
    )

    assert without_context.route == "E"
    assert with_context.route == "S"
    assert with_context.keyword_hits[0].match_policy == "cooccurrence"
    assert {hit.context_id for hit in with_context.keyword_hits[0].context_hits} >= {
        "CTX-DATA-OBJECT"
    }


def test_context_prefers_sentence_and_does_not_cross_boundary() -> None:
    same_sentence = matcher().match(
        record(
            claim="用户数据" + "甲" * 60 + "采用访问控制。",
            abstract="",
            title="",
            ipc="",
            main_ipc="",
        )
    )
    next_sentence = matcher().match(
        record(
            claim="用户数据。采用访问控制完成一般配置。",
            abstract="",
            title="",
            ipc="",
            main_ipc="",
        )
    )

    assert same_sentence.route == "S"
    assert same_sentence.keyword_hits[0].context_scope == "sentence"
    assert next_sentence.route == "E"


def test_context_uses_configured_window_without_sentence_boundaries() -> None:
    within = matcher().match(
        record(
            claim="用户数据" + "甲" * 40 + "访问控制",
            abstract="",
            title="",
            ipc="",
            main_ipc="",
        )
    )
    outside = matcher().match(
        record(
            claim="用户数据" + "甲" * 60 + "访问控制",
            abstract="",
            title="",
            ipc="",
            main_ipc="",
        )
    )

    assert within.route == "S"
    assert within.keyword_hits[0].context_scope == "window"
    assert outside.route == "E"


def test_ipc_is_audit_only_and_never_changes_route() -> None:
    result = matcher().match(
        record(claim="普通计算方法", abstract="", title="", ipc="G06F21/62", main_ipc="")
    )

    assert result.route == "E"
    assert result.ipc_audit_hits[0].rule_id == "IPC-AUDIT-G06F21"


def test_ascii_acronym_requires_token_boundaries() -> None:
    result = matcher().match(
        record(claim="a guaranteed service", abstract="", title="", ipc="", main_ipc="")
    )

    assert result.route == "E"


def test_ambiguous_dlp_acronym_requires_security_context() -> None:
    projection = matcher().match(
        record(claim="WIFI模式下的DLP控制系统", abstract="", title="", ipc="", main_ipc="")
    )
    leakage = matcher().match(
        record(claim="使用DLP防止数据泄露", abstract="", title="", ipc="", main_ipc="")
    )

    assert projection.route == "E"
    assert leakage.route == "S"
    assert any(hit.concept_id == "TECH-DLP-ACRONYM" for hit in leakage.keyword_hits)


def test_generic_monitoring_is_not_positive_without_security_meaning() -> None:
    result = matcher().match(
        record(claim="采集螺栓数据并监测拧紧状态", abstract="", title="", ipc="", main_ipc="")
    )

    assert result.route == "E"


def test_plaintext_requires_cryptographic_context() -> None:
    legal_phrase = matcher().match(
        record(claim="按照明文规定处理业务数据", abstract="", title="", ipc="", main_ipc="")
    )
    cryptographic = matcher().match(
        record(claim="将明文转换为密文", abstract="", title="", ipc="", main_ipc="")
    )

    assert legal_phrase.route == "E"
    assert cryptographic.route == "S"
    assert any(hit.concept_id == "TECH-PLAINTEXT" for hit in cryptographic.keyword_hits)


def test_repeated_context_terms_are_compacted_with_occurrence_count() -> None:
    result = matcher().match(
        record(
            claim="用户数据和用户数据采用访问控制",
            abstract="",
            title="",
            ipc="",
            main_ipc="",
        )
    )

    contexts = result.keyword_hits[0].context_hits
    specific = next(hit for hit in contexts if hit.context_id == "CTX-SPECIFIC-DATA-OBJECT")
    assert specific.matched_text == "用户数据"
    assert specific.occurrence_count == 2


def test_normalization_unifies_width_case_whitespace_and_connectors() -> None:
    assert normalize_text(" ＴＬＳ\tSecure—Channel ") == "tls secure-channel"
