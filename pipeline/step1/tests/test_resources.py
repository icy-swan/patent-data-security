from pipeline.step1.taxonomy import load_keyword_bundle


def test_resources_are_versioned_traceable_and_marked_as_pilot() -> None:
    bundle = load_keyword_bundle()
    source_ids = {source["id"] for source in bundle.sources["sources"]}

    assert bundle.keyword_version == "2.0.0-pilot.1"
    assert bundle.validation_protocol["status"] == "pilot_not_started"
    assert bundle.keywords["matching"]["ipc_changes_route"] is False
    assert {
        "METH-BESSEN-HUNT-2007",
        "METH-BENSON-MAGEE-2013",
        "METH-MOELLER-MOEHRLE-2015",
        "METH-XIE-MIYAZAKI-2013",
        "DOMAIN-ZHOU-2022",
    } <= source_ids
    assert set(bundle.hashes) == {
        "keywords",
        "sources",
        "validation_protocol",
        "changelog",
    }
