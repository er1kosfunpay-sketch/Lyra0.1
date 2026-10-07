"""Unit + end-to-end tests for the automated Russian evaluation report."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
from scripts import evaluate_russian as ev
from scripts.evaluate_russian import cyrillic_share, evaluate_answer, garbage_share, ngram_loop


def _tokenizer(tmp_path, vocab_size=512):
    """Random Cyrillic byte text -> a tokenizer whose vocabulary is exactly `vocab_size`."""
    import random

    rnd = random.Random(0)
    alphabet = "абвгдежзиклмнопрстуфхцчшщыьэюя"
    lines = []
    for _ in range(4000):
        word = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(3, 10)))
        lines.append(" ".join(
            "".join(rnd.choice(alphabet) for _ in range(rnd.randint(3, 10)))
            for _ in range(rnd.randint(3, 12))) + " " + word)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tok = LyraTokenizer.train([corpus], vocab_size=vocab_size)
    tok.save(tmp_path / "tok.json")
    return tok


def test_cyrillic_share_and_garbage():
    assert cyrillic_share("Привет, как дела?") == 1.0
    assert cyrillic_share("hello there") == 0.0
    assert cyrillic_share("") == 0.0
    assert garbage_share("Обычный русский текст.") == 0.0
    assert garbage_share("текст�ещё") > 0.0


def test_ngram_loop_detection():
    assert ngram_loop("да " * 40)
    assert ngram_loop("ну и ну и ну и ну и ну и ну и ну и ну и")
    assert not ngram_loop("Привет! Как у тебя дела сегодня? Я немного устал, но всё хорошо.")
    assert not ngram_loop("коротко")


def test_evaluate_answer_flags_bad_outputs(tmp_path):
    tok = _tokenizer(tmp_path)
    good = evaluate_answer("Привет", "Привет! Как дела?", tok.encode("Привет! Как дела?"), tok, 32)
    assert good["passed"], good
    assert good["checks"]["russian"] and good["checks"]["no_unk"]

    empty = evaluate_answer("Привет", "   ", [], tok, 32)
    assert not empty["checks"]["not_empty"] and not empty["passed"]

    english = evaluate_answer("Привет", "I am doing fine today, thanks", tok.encode("I am doing fine today, thanks"), tok, 32)
    assert not english["checks"]["russian"]

    unk_id = tok.id("<UNK>")
    unk = evaluate_answer("Привет", "ответ", [unk_id, unk_id], tok, 32)
    assert not unk["checks"]["no_unk"]

    leaked = evaluate_answer("Привет", "ответ <ASSISTANT> конец", tok.encode("ответ"), tok, 32)
    assert not leaked["checks"]["no_special_tokens"]
    assert "<ASSISTANT>" in leaked["special_tokens_leaked"]

    copy = evaluate_answer("Привет", "Привет", tok.encode("Привет"), tok, 32)
    assert not copy["checks"]["not_prompt_copy"]

    loop = evaluate_answer("Привет", "да " * 40, tok.encode("да " * 40), tok, 32)
    assert not loop["checks"]["no_repetition_loop"]


def test_evaluate_script_runs_end_to_end(tmp_path):
    """A tiny random model must still produce a complete PASS/FAIL report."""
    cfg = LyraConfig.from_json("configs/debug.json")
    tok = _tokenizer(tmp_path)
    tok_path = tmp_path / "tok.json"
    model = LyraModel(cfg)
    opt = torch.optim.AdamW(model.parameters())
    ckpt_path = tmp_path / "final.pt"
    from lyra.checkpoint import save_checkpoint

    save_checkpoint(ckpt_path, model, opt, cfg, step=3, tokens_seen=16,
                    tokenizer_fingerprint=hashlib.sha256(tok_path.read_bytes()).hexdigest(),
                    dataset_version="test")
    out_path = tmp_path / "report.json"
    rc = ev.main(["--checkpoint", str(ckpt_path), "--tokenizer", str(tok_path),
                  "--out", str(out_path), "--max-new-tokens", "6", "--no-multi-turn",
                  "--prompts", "Привет", "Как дела?"])
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert rc in (0, 1)
    assert report["overall"] in ("PASS", "FAIL")
    assert report["answers"] == 2
    assert report["step"] == 3
    assert report["model_parameters"] == cfg.parameter_count()
    assert set(report["check_pass_rates"]) >= {
        "not_empty", "no_unk", "no_special_tokens", "russian", "decode_clean",
        "no_repetition_loop", "not_prompt_copy", "reasonable_length"}
    for answer in report["single_turn"]:
        assert isinstance(answer["answer"], str)
        assert answer["cyrillic_share"] >= 0.0
    # A random, untrained model must not be graded as a Russian success.
    assert report["overall"] == "FAIL"


def test_vocab_mismatch_is_a_hard_error(tmp_path):
    """Tokenizer and model must agree on the vocabulary size."""
    cfg = LyraConfig.from_json("configs/debug.json")   # vocabulary 512
    _tokenizer(tmp_path, vocab_size=256)               # deliberately different
    ckpt_path = tmp_path / "final.pt"
    model = LyraModel(cfg)
    opt = torch.optim.AdamW(model.parameters())
    from lyra.checkpoint import save_checkpoint

    save_checkpoint(ckpt_path, model, opt, cfg, step=1, tokens_seen=1,
                    tokenizer_fingerprint=hashlib.sha256((tmp_path / "tok.json").read_bytes()).hexdigest(),
                    dataset_version="test")
    with pytest.raises(SystemExit):
        ev.load_model(ckpt_path, tmp_path / "tok.json", "cpu")
