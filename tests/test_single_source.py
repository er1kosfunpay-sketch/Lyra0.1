"""Guards: Lyra trains on exactly one dataset and nothing else may creep in."""
import json
from pathlib import Path

from scripts import prepare_data

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["scripts", "lyra", "configs", "kaggle"]
SCAN_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".ipynb", ".sh"}
FORBIDDEN = [
    "OpenAssistant", "oasst", "ultrachat", "dailydialog", "DailyDialog",
    "SiberianPersonaChat", "russian-everyday-dialogues", "Discord-Dialogues",
    "identity_and_honesty", "OpenHermes", "Tulu", "empathetic_dialogues",
    "lmsys-chat", "hh-rlhf", "russian_chat",
]
ALLOWED_FILES: set[str] = set()


def test_builder_declares_exactly_one_source():
    assert prepare_data.DATASET_REPO == "Den4ikAI/russian_dialogues_2"
    assert prepare_data.SOURCE_NAME == "ru_chat"
    assert prepare_data.ALLOWED_SOURCES == {"ru_chat"}


def _scan_files():
    for name in SCAN_DIRS:
        base = ROOT / name
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in ALLOWED_FILES:
                continue
            yield rel, path


def test_no_foreign_dataset_references_in_the_training_pipeline():
    offenders = []
    for rel, path in _scan_files():
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{rel}: {token}")
    assert not offenders, "Foreign dataset references found in the pipeline:\n" + "\n".join(offenders)


def test_prepare_data_status_declares_single_source(tmp_path, monkeypatch):
    from scripts import prepare_data

    raw = tmp_path / "raw.jsonl.gz"
    raw.write_bytes(b"x")
    monkeypatch.setattr(prepare_data, "resolve_raw_file", lambda refresh=False: (raw, "test"))
    monkeypatch.setattr(prepare_data, "iter_raw_records",
                        lambda path: iter([["Привет!", "Привет, как дела?"]]))
    out = tmp_path / "processed"
    assert prepare_data.main(["--out", str(out)]) == 0
    status = json.loads((out / "build_status.json").read_text(encoding="utf-8"))
    assert status["source"] == "ru_chat" and status["single_source"] is True
    assert status["fallback_used"] is False and status["failed_sources"] == []
