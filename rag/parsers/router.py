"""확장자 기반 파서 라우터 + 파일 단위 예외 격리.

[설계 의도]
- Pipeline은 "어떤 파서를 쓸지" 몰라도 되게 합니다. 새 포맷(.pptx 등) 추가 시
  파서 클래스를 만들고 `_parsers`에 등록만 하면 됩니다 (Open-Closed 원칙).
- `parse_many()`는 파일 하나가 실패해도 나머지를 계속 처리합니다. 1,000개 중
  1개의 손상 PDF 때문에 야간 배치 전체가 멈추는 것은 운영에서 가장 흔한 사고입니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rag.config import ParserConfig
from rag.exceptions import RAGError, UnsupportedFileError
from rag.parsers.base import BaseParser
from rag.parsers.docx_parser import DocxParser
from rag.parsers.excel_parser import ExcelParser
from rag.parsers.pdf_parser import PDFParser
from rag.schemas import ParsedDocument, ParseFailure

logger = logging.getLogger(__name__)


class ParserRouter:
    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        # 파서 인스턴스는 한 번만 생성해 재사용 (Docling 모델 로딩 비용 때문)
        parsers: List[BaseParser] = [
            PDFParser(self.config),
            ExcelParser(self.config),
            DocxParser(self.config),
        ]
        self._parsers: Dict[str, BaseParser] = {}
        for parser in parsers:
            for ext in parser.supported_extensions:
                self._parsers[ext] = parser

    @property
    def supported_extensions(self) -> List[str]:
        return sorted(self._parsers)

    def is_supported(self, path: str | Path) -> bool:
        p = Path(path)
        return p.suffix.lower() in self._parsers and not p.name.startswith("~$")

    def parse(self, path: str | Path) -> ParsedDocument:
        """단일 파일 파싱. 실패 시 도메인 예외(RAGError 하위)를 그대로 던집니다."""
        file_path = Path(path)
        parser = self._parsers.get(file_path.suffix.lower())
        if parser is None:
            raise UnsupportedFileError(
                f"지원하지 않는 확장자: {file_path.suffix or '(없음)'}", source=file_path
            )
        return parser.parse(file_path)

    def parse_many(
        self, paths: Iterable[str | Path]
    ) -> Tuple[List[ParsedDocument], List[ParseFailure]]:
        """여러 파일을 파싱하고 (성공 목록, 실패 목록)을 반환합니다. 절대 중간에 중단하지 않습니다."""
        documents: List[ParsedDocument] = []
        failures: List[ParseFailure] = []
        for path in paths:
            try:
                documents.append(self.parse(path))
            except RAGError as exc:
                # 예상된 실패(빈 문서, 암호, 손상): 경고 수준으로 기록하고 계속
                logger.warning("파싱 건너뜀: %s", exc)
                failures.append(ParseFailure(str(path), type(exc).__name__, exc.reason))
            except Exception as exc:  # noqa: BLE001
                # 예상하지 못한 실패: 스택 트레이스를 남겨 버그 추적이 가능하게 합니다.
                logger.exception("파싱 중 예기치 못한 오류: %s", path)
                failures.append(ParseFailure(str(path), type(exc).__name__, str(exc)))
        logger.info("파싱 결과: 성공 %d건, 실패 %d건", len(documents), len(failures))
        return documents, failures
