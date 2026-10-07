import json

import pytest

from scripts import prepare_data
from scripts.prepare_data import SOURCE_NAME, SourceUnavailableError, clean_sample


def sample(*utterances):
    return list(utterances)


def test_pipeline_is_single_source_by_name():
    assert prepare_data.DATASET_REPO == "Den4ikAI/russian_dialogues_2"
    assert prepare_data.ALLOWED_SOURCES == frozenset({"ru_chat"})
    assert prepare_data.SOURCE_NAME == "ru_chat"


def test_multi_source_request_is_rejected_with_clear_error(tmp_path):
    with pytest.raises(SystemExit):
        prepare_data.main(["--sources", "oasst,ru_chat", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        prepare_data.main(["--sources", "ru_chat,ultra", "--out", str(tmp_path)])


def test_clean_sample_keeps_normal_russian_dialogue():
    row, reason = clean_sample(sample("Привет, как дела?", "Привет! Нормально, спасибо."))
    assert reason is None
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert row["source"] == "ru_chat" and row["lang"] == "ru"


def test_clean_sample_rejects_broken_rows():
    assert clean_sample(sample("Привет"))[1] == "too_few_messages"
    assert clean_sample(["Привет", None])[1] == "non_string_message"
    assert clean_sample(["Привет", "мусор� битый текст"])[1] == "encoding_damage"
    assert clean_sample(["Привет", "   "])[1] == "empty_message_after_clean"
    assert clean_sample(["https://example.com", "https://example.org/page"])[1] == "link_only_message"
    assert clean_sample(["Hello my friend", "How are you doing today"])[1] == "not_russian"
    assert clean_sample(["а" * 40, "б" * 40])[1] == "meaningless_repetition"


def test_clean_sample_accepts_uppercase_cyrillic():
    """Russian written in CAPS is still Russian (the check is case-insensitive)."""
    row, reason = clean_sample(["ВАНИШ ОТ ВСЕХ ВИДОВ ПЯТЕН", "ВСЁ ЧИСТО"])
    assert reason is None, reason
    assert row["lang"] == "ru"
    # ...while Latin-CAPS text is still not Russian.
    assert clean_sample(["BRUSH ALL KINDS OF STAINS", "EVERYTHING IS CLEAN"])[1] == "not_russian"


def build(tmp_path, monkeypatch, samples, *extra):
    raw = tmp_path / "raw.jsonl.gz"
    raw.write_bytes(b"x")
    monkeypatch.setattr(prepare_data, "resolve_raw_file", lambda refresh=False: (raw, "test"))
    monkeypatch.setattr(prepare_data, "iter_raw_records", lambda path: iter(samples))
    out = tmp_path / "processed"
    assert prepare_data.main(["--out", str(out), *extra]) == 0
    return out


def read_split(out, name):
    path = out / f"{name}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_build_writes_all_splits_and_single_source_stats(tmp_path, monkeypatch):
    samples = [
        sample("Привет, как дела?", "Отлично, спасибо!"),
        sample("Чем занимаешься?", "Пишу код для своей модели."),
        sample("Какая погода?", "Сегодня солнечно и тепло."),
        sample("Привет, как дела?", "Отлично, спасибо!"),  # exact duplicate
        sample("Привет"),                                  # broken row
    ]
    out = build(tmp_path, monkeypatch, samples)
    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    status = json.loads((out / "build_status.json").read_text(encoding="utf-8"))

    assert stats["source_names"] == ["ru_chat"]
    assert stats["dataset"]["repo"] == "Den4ikAI/russian_dialogues_2"
    assert stats["single_source"] is True and stats["fallback_used"] is False
    assert stats["original_dialogues"] == 5
    assert stats["duplicates_removed"] == 1
    assert stats["rejections_by_reason"] == {"too_few_messages": 1}
    assert stats["clean_dialogues"] == 3
    assert stats["removed_total"] == 2
    total = sum(stats["splits"][name]["conversations"] for name in ("train", "validation", "test"))
    assert total == 3
    assert sum(stats["splits"][name]["messages"] for name in ("train", "validation", "test")) == 6
    assert status["build_state"] == "SUCCESS" and status["failed_sources"] == []
    for name in ("train", "validation", "test"):
        assert (out / f"{name}.jsonl").is_file()
    assert stats["limit"] == 0 and stats["limit_note"] is None


def test_failed_source_never_falls_back_to_another_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "resolve_raw_file",
                        lambda refresh=False: (_ for _ in ()).throw(SourceUnavailableError("offline")))
    out = tmp_path / "processed"
    assert prepare_data.main(["--out", str(out)]) == 2
    status = json.loads((out / "build_status.json").read_text(encoding="utf-8"))
    assert status["build_state"] == "FAILED_SOURCE_UNAVAILABLE"
    assert status["fallback_used"] is False and status["failed_sources"] == ["ru_chat"]
    assert not (out / "train.jsonl").exists()


def test_completed_build_is_reused_and_force_rebuilds(tmp_path, monkeypatch):
    samples = [sample("Привет!", "Привет, как дела?")]
    out = build(tmp_path, monkeypatch, samples)
    # Second run with the same parameters reuses the published splits.
    monkeypatch.setattr(prepare_data, "iter_raw_records",
                        lambda path: (_ for _ in ()).throw(AssertionError("rebuilt without --force")))
    assert prepare_data.main(["--out", str(out)]) == 0
    # --force runs the pipeline again.
    monkeypatch.setattr(prepare_data, "iter_raw_records", lambda path: iter(samples))
    assert prepare_data.main(["--out", str(out), "--force"]) == 0


def test_debug_limit_is_recorded_in_stats(tmp_path, monkeypatch):
    samples = [sample(f"Привет номер {i}", f"Ответ номер {i}") for i in range(10)]
    out = build(tmp_path, monkeypatch, samples, "--limit", "4")
    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    assert stats["clean_dialogues"] == 4
    assert stats["limit"] == 4 and "DEBUG LIMIT" in stats["limit_note"]
