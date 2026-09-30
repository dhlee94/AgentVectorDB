"""파서/청커 설정값.

[설계 의도]
- 청크 크기, 대용량 PDF 기준 같은 값은 문서 코퍼스마다 최적값이 다릅니다.
  코드 곳곳에 숫자를 하드코딩하면 튜닝할 때마다 코드를 고쳐야 하므로,
  모든 "조절 가능한 숫자"를 이 파일의 dataclass로 모았습니다.
- 운영 설정은 프로젝트 루트의 `config.yaml` 한 파일에서 관리합니다. 이 파일의 dataclass
  기본값은 "config.yaml에 항목이 없을 때"의 기본값이며, `load_config()`가 YAML을 덮어씁니다.
- API 키 같은 비밀값은 설정 파일에 넣지 않고 `.env`(git 제외)에 둡니다. `load_env()`가 환경변수로 올립니다.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rag.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    # ------------------------------------------------------------------ PDF
    # 이 페이지 수를 넘으면 Docling 대신 PyMuPDF4LLM으로 바로 보냅니다.
    # Docling은 페이지당 수 초가 걸릴 수 있어, 수백 페이지 매뉴얼을 Docling에 넣으면
    # 인제스트 전체가 몇 시간씩 막힙니다. "정확도 vs 처리량"의 경계값입니다.
    large_pdf_page_threshold: int = 300
    large_pdf_size_mb: float = 50.0

    # Docling 사용 여부. 설치가 무거운(torch 의존) 라이브러리라 끌 수 있게 둡니다.
    use_docling: bool = True
    # Docling 문서 1건당 최대 처리 시간(초). 지원하는 버전에서만 적용됩니다.
    docling_timeout_sec: float = 600.0

    # 사전 검사에서 페이지당 평균 텍스트가 이 글자 수 미만이면 스캔 PDF로 간주해 OCR을 켭니다.
    scanned_chars_per_page: int = 50
    # 사전 검사에 사용할 샘플 페이지 수 (전 페이지를 읽으면 사전 검사 자체가 느려짐).
    preflight_sample_pages: int = 10
    # (Docling 결과 글자 수 / PyMuPDF 원시 텍스트 글자 수)가 이 비율 미만이면
    # Docling이 본문을 누락했다고 보고 Fallback 파서로 재파싱합니다.
    min_docling_text_ratio: float = 0.5
    # PDF 헤더가 한 수준(##)으로 평평하게 추출되면 "N장/부록/1." 번호 패턴으로 계층을 복원합니다.
    # (이미 계층이 살아 있는 PDF에는 자동으로 적용되지 않습니다)
    pdf_restore_heading_levels: bool = True
    # 고정폭 글꼴로 조판된 숫자·기호가 `10`, `→`처럼 인라인 코드로 추출되면 백틱을 벗겨냅니다.
    pdf_strip_symbolic_inline_code: bool = True

    # ---------------------------------------------------------------- Excel
    # True면 병합 셀 유무와 관계없이 모든 시트를 서술형 템플릿으로 변환합니다.
    excel_force_narrative: bool = False
    # 다단 헤더로 인정할 최대 행 수.
    excel_max_header_rows: int = 3
    # 숨김 시트 포함 여부. 숨김 시트는 대개 VLOOKUP용 코드표라 검색 노이즈가 됩니다.
    excel_include_hidden_sheets: bool = False


@dataclass
class ChunkerConfig:
    # 1차 분할 기준 헤더. "####" 이하는 섹션 내부 소제목으로 보고 분할하지 않습니다.
    # (너무 깊은 헤더까지 자르면 Parent가 한두 문장짜리로 잘게 쪼개져 Small-to-Big 효과가 사라짐)
    headers_to_split_on: Tuple[Tuple[str, str], ...] = (
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    )

    # Parent 청크 크기(글자 수). 한국어는 토큰 수보다 글자 수가 예측 가능성이 높아 글자 기준을 씁니다.
    parent_max_chars: int = 3000
    # 이보다 작은 섹션은 같은 장(최상위 헤더) 안의 인접 섹션과 병합합니다.
    parent_min_chars: int = 800

    # Child 청크 크기와 겹침. 태그 길이는 이 예산 안에서 차감됩니다.
    child_chunk_chars: int = 500
    child_overlap_chars: int = 50
    # 태그를 빼고 남는 본문 예산의 하한. 섹션명이 비정상적으로 긴 경우의 안전장치입니다.
    min_child_body_chars: int = 150
    # 공백 제거 후 이 글자 수 미만인 Child는 버립니다 (페이지 번호만 남은 조각 등).
    min_child_chars: int = 10

    # 표 바로 앞 문단이 이 길이 이하면 "표 제목(캡션)"으로 보고 잘린 표 조각마다 함께 붙입니다.
    table_caption_max_chars: int = 150

    # 헤더가 전혀 없는 문서의 섹션명.
    default_section: str = "본문"
    # 여러 소섹션을 병합한 Parent의 섹션 라벨에 나열할 최대 소섹션 수.
    max_section_label_items: int = 3


@dataclass
class StoreConfig:
    # Chroma(Child 벡터), Docstore(Parent), 매니페스트가 모두 이 디렉터리 아래에 저장됩니다.
    persist_dir: str = "./store"
    collection_name: str = "rag_children"
    # 임베딩 모델 이름은 매니페스트에 기록됩니다. 다른 모델로 만든 벡터가 한 컬렉션에 섞이면
    # 에러 없이 검색 품질만 망가지므로, 모델이 바뀌면 재인덱싱을 강제합니다.
    embedding_model: str = "BAAI/bge-m3"
    # None이면 sentence-transformers가 자동 선택 (cuda > mps > cpu)
    embedding_device: Optional[str] = None
    # Chroma 1회 add 최대 개수. 너무 크면 Chroma 배치 한도 초과, 너무 작으면 임베딩 효율 저하.
    add_batch_size: int = 256


@dataclass
class RetrieverConfig:
    # 각 검색기에서 가져올 Child 후보 수. Parent top_n보다 충분히 커야 RRF 융합 효과가 납니다.
    dense_k: int = 20
    sparse_k: int = 20
    # RRF 상수. 60은 원 논문(Cormack et al., 2009) 값으로, 상위 몇 개 순위 차이를 완만하게 만듭니다.
    rrf_k: int = 60
    # 제품코드·수치 질의가 많은 코퍼스면 sparse_weight를 올립니다.
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    # LLM에 전달할 Parent 수
    top_n: int = 5


@dataclass
class GenerationConfig:
    # Claude Haiku 4.5: 빠르고 저렴한 모델. 품질이 더 필요하면 "claude-sonnet-5-5" / "claude-opus-5-5".
    # 모델을 바꾸면 아래 effort / temperature / enable_refusal_fallback도 모델에 맞게 바꿔야 합니다.
    # (잘못된 조합은 validate()가 설정 로드 시점에 알려 줍니다)
    model: str = "claude-haiku-4-5"
    # 사고 깊이/토큰 사용량 조절. Opus 4.5+, Sonnet 4.6+, Sonnet 5.x, Fable 전용.
    # Haiku 4.5에 보내면 400 오류이므로 None(미전송)입니다.
    effort: Optional[str] = None
    # 0이면 같은 질문에 최대한 같은 답 → 사실 기반 RAG에 적합합니다.
    # Opus 4.7+ / Opus 5.x / Sonnet 5.x / Fable은 temperature를 거부(400)하므로 None으로 두세요.
    temperature: Optional[float] = 0.0
    # 비스트리밍 호출이라 HTTP 타임아웃을 피하도록 16K 이하를 권장합니다 (Haiku 4.5 최대 64K).
    max_tokens: int = 4096
    # SDK가 429/5xx/연결 오류를 지수 백오프로 자동 재시도합니다.
    max_retries: int = 3
    timeout_sec: float = 120.0
    # 안전 분류기 거절 시 서버가 다른 모델로 자동 재시도 (Opus 5.5 / Opus 5 / Sonnet 5.5 / Fable 5.1 전용 베타).
    enable_refusal_fallback: bool = False
    # LLM에 넣을 컨텍스트(Parent 본문 합계) 최대 글자 수. Haiku 4.5 컨텍스트는 200K 토큰이라 여유가 큽니다.
    max_context_chars: int = 24000


@dataclass
class PipelineConfig:
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # ------------------------------------------------------------ 파일 로딩
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """YAML 설정 파일을 읽어 기본값 위에 덮어씁니다. 파일에 없는 항목은 코드 기본값을 씁니다.

        오타 방지를 위해 모르는 키가 있으면 조용히 무시하지 않고 ConfigError를 냅니다.
        (예: `child_chunk_size` 오타를 무시하면 "설정을 바꿨는데 반영이 안 된다"는 긴 디버깅이 시작됩니다)
        """
        import yaml

        file_path = Path(path).expanduser()
        try:
            raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise ConfigError("설정 파일이 없습니다", source=file_path) from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML 문법 오류: {exc}", source=file_path) from exc
        if not isinstance(raw, dict):
            raise ConfigError("설정 파일 최상위는 key: value 형태여야 합니다", source=file_path)

        config = cls()
        _apply_overrides(config, raw, prefix="", source=file_path)
        config.validate()
        return config

    def validate(self) -> None:
        """값의 범위와 모델별 파라미터 호환성을 검사합니다 (첫 질의가 400으로 실패하기 전에 조기 차단)."""
        errors: List[str] = []
        c, r, g = self.chunker, self.retriever, self.generation

        if c.child_chunk_chars >= c.parent_max_chars:
            errors.append("chunker.child_chunk_chars는 parent_max_chars보다 작아야 합니다")
        if c.parent_min_chars > c.parent_max_chars:
            errors.append("chunker.parent_min_chars는 parent_max_chars 이하여야 합니다")
        if c.child_overlap_chars >= c.child_chunk_chars:
            errors.append("chunker.child_overlap_chars는 child_chunk_chars보다 작아야 합니다")
        if min(r.dense_k, r.sparse_k, r.top_n, r.rrf_k) <= 0:
            errors.append("retriever의 dense_k / sparse_k / top_n / rrf_k는 양수여야 합니다")
        if r.dense_weight < 0 or r.sparse_weight < 0 or r.dense_weight + r.sparse_weight == 0:
            errors.append("retriever 가중치는 0 이상이고 합이 0보다 커야 합니다")
        if g.max_tokens <= 0:
            errors.append("generation.max_tokens는 양수여야 합니다")
        errors.extend(_model_compat_errors(g))

        if errors:
            raise ConfigError("설정 오류:\n  - " + "\n  - ".join(errors))


_PLACEHOLDER_KEYS = ("", "sk-ant-...", "your-api-key")


def load_env(dotenv_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """`.env` 파일을 읽어 환경변수로 등록합니다. 읽은 파일 경로(없으면 None)를 반환합니다.

    - 이미 설정된 환경변수는 덮어쓰지 않습니다(override=False). 서버/CI에서 주입한 실제 값이
      개발자 PC용 .env에 의해 바뀌는 사고를 막기 위함입니다.
    - 현재 작업 디렉터리부터 상위로 올라가며 .env를 찾으므로, 하위 폴더에서 실행해도 동작합니다.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        logger.warning("python-dotenv 미설치 → .env를 읽지 않습니다 (`pip install python-dotenv`)")
        return None

    path = str(dotenv_path) if dotenv_path else find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)

    # .env.example을 복사만 하고 키를 안 바꾼 경우, 빈 값/예시 값이 "설정된 키"로 취급되어
    # 인증 오류 원인을 찾기 어려워집니다. 이런 값은 지우고 명확히 경고합니다.
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key is not None and key.strip() in _PLACEHOLDER_KEYS:
        del os.environ["ANTHROPIC_API_KEY"]
        logger.warning("ANTHROPIC_API_KEY가 비어 있거나 예시 값입니다 — .env에 실제 키를 입력하세요")
    return path or None


def load_config(path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """설정 로드 우선순위: 인자로 준 경로 → 환경변수 RAG_CONFIG → ./config.yaml → 코드 기본값.

    .env를 먼저 읽으므로 RAG_CONFIG도 .env에 적어 둘 수 있습니다.
    """
    load_env()
    candidate = path or os.environ.get("RAG_CONFIG")
    if candidate:
        return PipelineConfig.from_yaml(candidate)
    if Path(DEFAULT_CONFIG_FILE).exists():
        return PipelineConfig.from_yaml(DEFAULT_CONFIG_FILE)
    config = PipelineConfig()
    config.validate()
    return config


DEFAULT_CONFIG_FILE = "config.yaml"


# ---------------------------------------------------------------------------
# 모델별 파라미터 호환성 (2026-09 기준). 목록은 조기 경고용이며, 최종 판단은 API 응답입니다.
# ---------------------------------------------------------------------------
_RETIRED_PREFIXES = ("claude-3", "claude-2", "claude-instant")
_EFFORT_UNSUPPORTED_PREFIXES = ("claude-haiku", "claude-sonnet-4-5", "claude-sonnet-4-0", "claude-opus-4-0", "claude-opus-4-1")
_TEMPERATURE_UNSUPPORTED_PREFIXES = (
    "claude-opus-5", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5", "claude-fable", "claude-mythos",
)
_FALLBACK_SUPPORTED = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1")
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _model_compat_errors(g: GenerationConfig) -> List[str]:
    errors: List[str] = []
    model = g.model.strip()
    if model.startswith(_RETIRED_PREFIXES):
        errors.append(
            f"generation.model '{model}'은(는) 서비스가 종료된 모델입니다 (호출 시 404). "
            "저렴한 모델이 필요하면 'claude-haiku-4-5'를 사용하세요"
        )
    if g.effort is not None:
        if g.effort not in _EFFORT_LEVELS:
            errors.append(f"generation.effort는 {_EFFORT_LEVELS} 중 하나 또는 null이어야 합니다")
        elif model.startswith(_EFFORT_UNSUPPORTED_PREFIXES):
            errors.append(f"'{model}'은(는) effort를 지원하지 않습니다 → generation.effort: null")
    if g.temperature is not None:
        if not 0.0 <= g.temperature <= 1.0:
            errors.append("generation.temperature는 0.0~1.0 사이여야 합니다")
        elif model.startswith(_TEMPERATURE_UNSUPPORTED_PREFIXES):
            errors.append(f"'{model}'은(는) temperature를 지원하지 않습니다 → generation.temperature: null")
    if g.enable_refusal_fallback and model not in _FALLBACK_SUPPORTED:
        errors.append(
            f"'{model}'은(는) 서버 측 거절 폴백을 지원하지 않습니다 → generation.enable_refusal_fallback: false"
        )
    return errors


def _apply_overrides(target: Any, values: Dict[str, Any], prefix: str, source: Path) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        dotted = f"{prefix}{key}"
        if key not in known:
            raise ConfigError(f"알 수 없는 설정 키: '{dotted}' (가능한 키: {', '.join(known)})", source=source)
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"'{dotted}'는 하위 항목(key: value)이 있는 섹션이어야 합니다", source=source)
            _apply_overrides(current, value, prefix=f"{dotted}.", source=source)
        elif key == "headers_to_split_on":
            # YAML 리스트 [["#", "h1"], ...] → 청커가 기대하는 튜플의 튜플
            setattr(target, key, tuple(tuple(pair) for pair in value))
        elif isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            setattr(target, key, float(value))  # YAML의 "300"(int)을 float 필드에 허용
        else:
            setattr(target, key, value)
