import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docparse import llm as llm_mod
from docparse.llm import GeminiLLM, LLMUnavailable

ROOT = Path(__file__).parent.parent


def test_gemini_request_shape_and_response_parsing(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}).encode()

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"], seen["key"] = req.full_url, json.loads(req.data), req.get_header("X-goog-api-key")
        return Resp()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]
    assert GeminiLLM("m").complete(msgs) == "{}"
    assert seen["url"].endswith("/models/m:generateContent") and seen["key"] == "test-key"
    assert "test-key" not in seen["url"]  # ключ только в заголовке
    b = seen["body"]
    assert b["systemInstruction"]["parts"][0]["text"] == "S"
    assert [c["role"] for c in b["contents"]] == ["user", "model"]
    assert b["generationConfig"]["temperature"] == 0 and "properties" in b["generationConfig"]["responseJsonSchema"]


def test_missing_api_key_raises_dedicated_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        GeminiLLM().complete([])


def test_cli_exit_code_4_without_api_key():
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
    p = subprocess.run([sys.executable, "-m", "docparse", "data/docs/01.txt"], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", env=env)
    assert p.returncode == 4 and "GOOGLE_API_KEY" in p.stderr


def test_replay_eval_runs_offline_without_key_and_has_no_hallucinations():
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
    p = subprocess.run([sys.executable, "eval.py", "--replay"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", env=env)
    assert p.returncode == 0, p.stderr
    assert "ГАЛЛЮЦИНАЦИИ (в эталоне null, заполнено): 0" in p.stdout and "(replay)" in p.stdout
