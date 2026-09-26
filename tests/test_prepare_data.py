import json

from scripts import prepare_data


def conversation(lang, suffix):
    if lang == "ru":
        user, assistant = f"Привет, расскажи новость {suffix}?", f"Сегодня случилось интересное событие номер {suffix}."
    else:
        user, assistant = f"Hello, tell me a story {suffix}?", f"Here is an interesting story number {suffix}."
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], "lang": lang}


def test_auto_balance_keeps_all_and_fixed_never_oversamples():
    rows = [conversation("ru", i) for i in range(4)] + [conversation("en", i) for i in range(10)]
    auto = prepare_data.balance_rows(rows, "auto", None, seed=4)
    fixed = prepare_data.balance_rows(rows, "fixed", 0.5, seed=4)
    assert len(auto) == 14
    assert len(fixed) == 8
    assert sum(r["lang"] == "ru" for r in fixed) == 4
    assert sum(r["lang"] == "en" for r in fixed) == 4
    assert len({prepare_data.duplicate_key(r["messages"]) for r in fixed}) == len(fixed)


def test_siberian_parser_strips_persona_header_and_recovers_turns():
    prompt = (
        "Persona boilerplate. Недавно, у меня был следующий диалог: "
        "Ты: Привет, как прошел день? Я: День прошел хорошо. "
        "Ты: Что интересного случилось? Я:"
    )
    messages = prepare_data.parse_siberian_dialogue(prompt, "Мы сходили в музей.", "woman")
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert "Persona boilerplate" not in " ".join(m["content"] for m in messages)
    assert messages[-1]["content"] == "Мы сходили в музей."
    assert prepare_data.parse_siberian_dialogue(prompt, "ответ", "qa") is None


def test_source_cache_is_atomic_and_reusable(tmp_path):
    source_rows = [conversation("ru", 1)]
    meta = prepare_data.write_source_cache(tmp_path, "oasst", source_rows, limit=10, candidates=1, rejected=0, source_duplicates=0)
    got, loaded_meta = prepare_data.read_complete_cache(tmp_path, "oasst", 10)
    assert got == source_rows
    assert loaded_meta["status"] == "SUCCESS"
    assert meta["accepted"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_optional_source_keeps_successful_source_and_partial_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fake_rows(source, limit):
        if source == "ru_everyday":
            for i in range(4):
                yield {**conversation("ru", i), "source": source}
        elif source == "ultra":
            raise OSError("simulated streaming parquet outage")

    monkeypatch.setattr(prepare_data, "rows", fake_rows)
    out = tmp_path / "processed"
    assert prepare_data.main(["--sources", "ru_everyday,ultra", "--out", str(out)]) == 0
    status = json.loads((out / "build_status.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    assert status["ru_everyday"] == "SUCCESS"
    assert status["ultra"] == "SOURCE_FAILED"
    assert status["build_state"] == "PARTIAL"
    assert stats["accepted_after_balance"] == 4
    total = sum(len((out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()) for name in ("train", "validation", "test"))
    assert total == 4
    assert "WARNING:" in capsys.readouterr().out

    def fail_during_refresh(source, limit):
        raise OSError("offline during explicit source refresh")

    monkeypatch.setattr(prepare_data, "rows", fail_during_refresh)
    assert prepare_data.main(["--sources", "ru_everyday,ultra", "--refresh-sources", "--out", str(out)]) == 0
    refreshed = json.loads((out / "build_status.json").read_text(encoding="utf-8"))
    assert refreshed["ru_everyday"] == "SOURCE_FAILED"
    assert refreshed["sources"]["ru_everyday"]["using_cached_data"] is True
    total_after_refresh = sum(len((out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()) for name in ("train", "validation", "test"))
    assert total_after_refresh == 4
