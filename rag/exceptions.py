"""파이프라인 전역 예외 계층.

[설계 의도]
- 운영에서 가장 중요한 것은 "어떤 파일이 왜 빠졌는가"를 추적하는 것입니다.
  `Exception` 하나로 뭉뚱그리면 파이프라인 상위(Pipeline.ingest)에서
  "건너뛰어도 되는 실패(빈 문서, 암호 PDF)"와 "즉시 중단해야 하는 실패(설정 오류)"를
  구분할 수 없습니다. 그래서 도메인 예외를 계층으로 나눕니다.
- 모든 예외는 `source`(파일 경로)를 들고 다니게 하여, 로그와 실패 리포트에
  별도 가공 없이 바로 출력할 수 있게 했습니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


class RAGError(Exception):
    """파이프라인에서 발생하는 모든 예외의 최상위 클래스."""

    def __init__(self, message: str, source: Optional[Union[str, Path]] = None) -> None:
        self.source = str(source) if source is not None else None
        # 로그 한 줄만 보고도 원인 파일을 알 수 있도록 메시지에 경로를 포함합니다.
        full_message = f"{message} (source={self.source})" if self.source else message
        super().__init__(full_message)
        self.reason = message


class ConfigError(RAGError):
    """설정 파일 오류 (오타 키, 범위 밖 값, 모델과 맞지 않는 파라미터)."""


# ---------------------------------------------------------------------------
# 파싱 단계 예외
# ---------------------------------------------------------------------------
class ParserError(RAGError):
    """파서가 문서를 읽지 못한 일반적인 실패 (손상 파일, 라이브러리 내부 오류 등)."""


class UnsupportedFileError(ParserError):
    """지원하지 않는 확장자이거나 처리 대상이 아닌 파일 (예: Office 잠금 파일 `~$*.xlsx`)."""


class EncryptedDocumentError(ParserError):
    """비밀번호가 필요한 문서. 재시도해도 성공할 수 없으므로 Fallback 대상이 아닙니다."""


class EmptyDocumentError(ParserError):
    """파싱은 성공했지만 의미 있는 텍스트가 전혀 없는 문서 (빈 파일, 이미지 전용 PDF 등)."""


# ---------------------------------------------------------------------------
# 청킹 단계 예외
# ---------------------------------------------------------------------------
class ChunkingError(RAGError):
    """청킹 결과가 비정상(청크 0개 등)이어서 인덱싱을 진행하면 안 되는 경우."""


# ---------------------------------------------------------------------------
# 인덱싱 / 검색 / 생성 단계 예외
# ---------------------------------------------------------------------------
class IndexingError(RAGError):
    """Vector DB/Docstore 쓰기 실패. 해당 문서는 롤백되고 매니페스트에 기록되지 않습니다."""


class IndexConfigMismatchError(IndexingError):
    """기존 인덱스와 현재 설정(임베딩 모델 등)이 달라 그대로 쓰면 검색 품질이 망가지는 경우."""


class RetrievalError(RAGError):
    """Dense/Sparse 검색이 모두 실패한 경우."""


class GenerationError(RAGError):
    """LLM 호출 실패. retryable=True면 잠시 후 재시도로 해결될 수 있는 오류입니다."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
