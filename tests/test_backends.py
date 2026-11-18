import json
import subprocess
import sys
from pathlib import Path

import pytest

from docparse import llm as llm_mod
from docparse.llm import LLMUnavailable, OllamaLLM

ROOT = Path(__file__).parent.parent


def test_ollama_request_shape_and_response_parsing(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"message": {"content": "{}"}}).encode()

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return Resp()

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    out = OllamaLLM(model="m", host="http://h:1/").complete([{"role": "user", "content": "x"}])
    assert out == "{}" and seen["url"] == "http://h:1/api/chat"
    assert seen["body"]["stream"] is False and seen["body"]["options"]["temperature"] == 0
    assert "properties" in seen["body"]["format"]  # JSON-схема ограничивает генерацию


def test_ollama_down_raises_dedicated_error():
    with pytest.raises(LLMUnavailable):
        OllamaLLM(host="http://127.0.0.1:9", timeout=1).complete([])


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "docparse", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def test_cli_prints_valid_json_for_a_document():
    p = run_cli("data/docs/01.txt")
    assert p.returncode == 0
    assert json.loads(p.stdout)["total_amount"] == 186180


def test_cli_exit_code_4_when_model_unavailable():
    p = subprocess.run([sys.executable, "-m", "docparse", "data/docs/01.txt", "--backend", "ollama"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", env={**__import__("os").environ, "OLLAMA_HOST": "http://127.0.0.1:9"})
    assert p.returncode == 4 and "error:" in p.stderr


def test_eval_heuristic_has_no_hallucinations():
    p = subprocess.run([sys.executable, "eval.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0 and "ГАЛЛЮЦИНАЦИИ (в эталоне null, заполнено): 0" in p.stdout
