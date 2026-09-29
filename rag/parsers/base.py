"""모든 파서의 추상 기반 클래스.

[설계 의도 — Template Method 패턴]
- 파일 검증 → 포맷별 파싱 → 정규화 → 빈 문서 검사 순서는 모든 포맷에서 동일합니다.
  이 공통 흐름을 `parse()`에 고정하고, 서브클래스는 `_parse()`(포맷별 핵심 로직)만
  구현하게 했습니다. 새 파서를 추가하는 개발자가 NFC 정규화나 빈 문서 검사를
  "깜빡하는" 실수를 구조적으로 막기 위함입니다.
"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, FrozenSet, List

from rag.config import ParserConfig
from rag.exceptions import EmptyDocumentError, ParserError, RAGError, UnsupportedFileError
from rag.markdown_utils import has_meaningful_content, normalize_text
from rag.schemas import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class ParseOutput:
    """서브클래스 `_parse()`의 반환값. 공통 필드(doc_id 등)는 베이스가 채웁니다."""

    markdown: str
    parser_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseParser(ABC):
    #: 이 파서가 처리하는 확장자 (소문자, 점 포함).
    supported_extensions: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    # ------------------------------------------------------------------ public
    def parse(self, path: str | Path) -> ParsedDocument:
        """파일 하나를 파싱해 `ParsedDocument`를 반환합니다.

        Raises:
            UnsupportedFileError, EncryptedDocumentError, EmptyDocumentError, ParserError
        """
        file_path = Path(path).expanduser().resolve()
        self._validate_file(file_path)

        try:
            output = self._parse(file_path)
        except RAGError:
            # 이미 도메인 예외로 분류된 것은 그대로 올려 보냅니다 (분류 정보 보존).
            raise
        except MemoryError as exc:
            # 거대한 파일에서 흔히 발생. 프로세스 전체를 죽이지 않도록 도메인 예외로 전환합니다.
            raise ParserError(f"메모리 부족으로 파싱 실패: {exc}", source=file_path) from exc
        except Exception as exc:  # noqa: BLE001 — 서드파티 파서는 어떤 예외든 던질 수 있음
            # 서드파티 라이브러리의 예외 타입을 상위 레이어까지 새어나가게 하면
            # Pipeline이 fitz/openpyxl/docling 예외를 전부 알아야 합니다. 여기서 차단합니다.
            raise ParserError(f"{type(exc).__name__}: {exc}", source=file_path) from exc

        markdown = normalize_text(output.markdown)
        if not has_meaningful_content(markdown):
            raise EmptyDocumentError("파싱 결과에 의미 있는 텍스트가 없습니다", source=file_path)

        # macOS의 NFD 파일명을 NFC로 통일 (normalize_text 주석 참고)
        source_name = unicodedata.normalize("NFC", file_path.name)
        doc = ParsedDocument(
            doc_id=self.make_doc_id(file_path),
            source=source_name,
            source_path=unicodedata.normalize("NFC", str(file_path)),
            file_type=file_path.suffix.lower().lstrip("."),
            markdown=markdown,
            parser_used=output.parser_used,
            metadata=output.metadata,
            warnings=output.warnings,
        )
        for warning in doc.warnings:
            logger.warning("[%s] %s", source_name, warning)
        logger.info(
            "파싱 완료: %s (parser=%s, %d chars)", source_name, doc.parser_used, len(markdown)
        )
        return doc

    @staticmethod
    def make_doc_id(file_path: Path) -> str:
        """경로 기반 결정적 ID.

        내용 해시가 아닌 경로 해시를 쓰는 이유: 같은 파일이 수정되어도 doc_id가 유지되어야
        "이 문서의 이전 청크를 지우고 새로 넣는다"는 갱신 로직이 가능합니다.
        내용 변경 감지는 Loader의 매니페스트(SHA-256)가 담당합니다.
        """
        normalized = unicodedata.normalize("NFC", str(file_path))
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    # --------------------------------------------------------------- template
    @abstractmethod
    def _parse(self, file_path: Path) -> ParseOutput:
        """포맷별 파싱 로직. 검증된 경로가 들어옵니다."""

    # ---------------------------------------------------------------- helpers
    def _validate_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise ParserError("파일이 존재하지 않습니다", source=file_path)
        if not file_path.is_file():
            raise UnsupportedFileError("일반 파일이 아닙니다", source=file_path)
        # Office가 파일을 열고 있는 동안 만드는 잠금 파일. 실제 문서가 아니라 수백 바이트짜리
        # 메타데이터라 파싱하면 반드시 실패합니다. 공유 폴더 운영 시 매우 자주 마주칩니다.
        if file_path.name.startswith("~$"):
            raise UnsupportedFileError("Office 잠금 파일은 처리하지 않습니다", source=file_path)
        if file_path.suffix.lower() not in self.supported_extensions:
            raise UnsupportedFileError(
                f"{type(self).__name__}가 지원하지 않는 확장자: {file_path.suffix}",
                source=file_path,
            )
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            raise ParserError(f"파일 정보를 읽을 수 없습니다: {exc}", source=file_path) from exc
        if size == 0:
            raise EmptyDocumentError("0바이트 파일입니다", source=file_path)
