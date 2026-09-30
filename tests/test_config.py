"""설정: config.yaml ↔ 코드 기본값 일치, 오타·모델 호환성 검사, .env 로딩."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from rag.config import PipelineConfig, load_config, load_env
from rag.exceptions import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_repo_config_yaml_matches_code_defaults():
    # config.yaml과 dataclass 기본값이 어긋나면 "파일을 안 쓰면 다른 설정"이 되는 혼란이 생김
    loaded = PipelineConfig.from_yaml(REPO_CONFIG)
    assert dataclasses.asdict(loaded) == dataclasses.asdict(PipelineConfig())


def _write(tmp_path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        ("chunker:\n  child_chunk_size: 400\n", "알 수 없는 설정 키"),
        ("generation:\n  model: claude-3-haiku-20240307\n", "서비스가 종료된 모델"),
        ("generation:\n  effort: high\n", "effort를 지원하지 않습니다"),
        ("generation:\n  model: claude-opus-5-5\n", "temperature를 지원하지 않습니다"),
        ("generation:\n  enable_refusal_fallback: true\n", "거절 폴백을 지원하지 않습니다"),
        ("chunker:\n  child_chunk_chars: 5000\n", "parent_max_chars보다 작아야"),
        ("retriever:\n  top_n: 0\n", "양수여야"),
        ("generation: 3\n", "섹션이어야"),
    ],
)
def test_invalid_configs_rejected(tmp_path, yaml_text, message):
    with pytest.raises(ConfigError, match=message):
        PipelineConfig.from_yaml(_write(tmp_path, yaml_text))


def test_valid_override_for_opus(tmp_path):
    cfg = PipelineConfig.from_yaml(
        _write(
            tmp_path,
            "generation:\n  model: claude-opus-5-5\n  effort: high\n  temperature: null\n"
            "  enable_refusal_fallback: true\nretriever:\n  top_n: 3\n",
        )
    )
    assert (cfg.generation.model, cfg.generation.effort, cfg.retriever.top_n) == ("claude-opus-5-5", "high", 3)
    assert cfg.chunker.parent_max_chars == 3000  # 파일에 없는 항목은 기본값


def test_load_config_uses_rag_config_env(tmp_path, monkeypatch):
    path = _write(tmp_path, "retriever:\n  top_n: 7\n")
    monkeypatch.setenv("RAG_CONFIG", str(path))
    assert load_config().retriever.top_n == 7


def test_load_config_falls_back_to_defaults(tmp_path):
    # isolate_from_repo 픽스처로 cwd가 빈 임시 디렉터리 → config.yaml 없음
    assert dataclasses.asdict(load_config()) == dataclasses.asdict(PipelineConfig())


# ------------------------------------------------------------------ .env
def test_load_env_reads_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-test-123\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")  # 하위 폴더에서도 상위의 .env를 찾아야 함
    load_env()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-123"


def test_shell_env_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8")
    load_env()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"


def test_placeholder_key_is_removed(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-...\n", encoding="utf-8")
    load_env()
    assert "ANTHROPIC_API_KEY" not in os.environ
