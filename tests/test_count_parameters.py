"""The parameter-count tool must agree with the real model, for every profile."""
import json
from pathlib import Path

from lyra.config import LyraConfig
from lyra.model import LyraModel
from scripts.count_parameters import compute_param_count, main

ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return sorted((ROOT / "configs").glob("*.json"))


def test_every_profile_has_a_valid_model_config():
    assert _configs(), "no model configs found"
    for path in _configs():
        cfg = LyraConfig.from_json(str(path))
        assert cfg.vocab_size > 0 and cfg.num_layers > 0


def test_formula_matches_config_parameter_count():
    for path in _configs():
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        cfg = LyraConfig.from_json(str(path))
        assert compute_param_count(raw) == cfg.parameter_count(), path.name


def test_formula_matches_the_instantiated_model():
    cfg = LyraConfig.from_json(str(ROOT / "configs" / "debug.json"))
    model = LyraModel(cfg)
    assert compute_param_count(cfg.to_dict()) == model.parameter_count()
    assert cfg.parameter_count() == model.parameter_count()


def test_main_reports_counts_and_accepts_a_target(capsys):
    assert main(["--config", str(ROOT / "configs" / "debug.json")]) == 0
    out = capsys.readouterr().out
    assert "Parameters (formula):" in out
    assert "Parameters (model.parameter_count()):" in out

    assert main(["--config", str(ROOT / "configs" / "debug.json"),
                 "--target", "1"]) == 0
    assert "Difference:" in capsys.readouterr().out


def test_main_fails_cleanly_on_a_missing_config(tmp_path):
    assert main(["--config", str(tmp_path / "nope.json")]) == 1
