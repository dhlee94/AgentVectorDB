"""Word(.docx) 파서: Docling → python-docx Fallback.

[설계 의도]
- Docling은 DOCX의 제목 스타일을 `#` 헤더로, 표를 Markdown 표로 정확히 옮겨 주므로 1순위입니다.
- python-docx Fallback은 "본문 순서 보존"이 핵심입니다. `document.paragraphs`와
  `document.tables`를 따로 순회하면 표가 전부 문서 끝으로 몰려 "어느 절에 속한 표인지"라는
  문맥이 사라집니다. 그래서 body XML의 자식 노드를 원래 순서대로 순회합니다.
- 머리글/바닥글은 의도적으로 제외합니다. 모든 페이지에 반복되는 회사명·문서번호가
  청크마다 섞여 들어가 검색 노이즈만 늘리기 때문입니다.
- DOCX는 페이지가 렌더링 환경에 따라 달라지므로 페이지 마커를 넣지 않습니다(섹션으로 인용).
"""

from __future__ import annotations

import importlib.util
import logging
import re
import threading
import zipfile
from pathlib import Path
from typing import Any, List, Optional

from rag.config import ParserConfig
from rag.exceptions import EncryptedDocumentError, ParserError
from rag.markdown_utils import build_markdown_table
from rag.parsers.base import BaseParser, ParseOutput

logger = logging.getLogger(__name__)

_OLE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")
# 영문/한글 Word 모두 대응: "Heading 2", "heading 2", "제목 2"
_HEADING_STYLE_RE = re.compile(r"^(?:heading|제목)\s*(\d)$", re.IGNORECASE)


class DocxParser(BaseParser):
    supported_extensions = frozenset({".docx"})

    _docling_converter: Any = None
    _docling_lock = threading.Lock()

    def __init__(self, config: ParserConfig | None = None) -> None:
        super().__init__(config)
        self._docling_available = self.config.use_docling and self._check_docling_installed()

    # ------------------------------------------------------------------ main
    def _parse(self, file_path: Path) -> ParseOutput:
        with file_path.open("rb") as f:
            if f.read(8) == _OLE_SIGNATURE:
                # 암호화된 DOCX는 OLE 컨테이너입니다. (구형 .doc를 .docx로 개명한 경우도 여기에 걸림)
                raise EncryptedDocumentError(
                    "암호가 설정되었거나 구형 .doc 형식인 Word 파일입니다", source=file_path
                )

        warnings: List[str] = []
        if self._docling_available:
            try:
                markdown = self._parse_with_docling(file_path)
                return ParseOutput(markdown, "docling", {}, warnings)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Docling 실패 → python-docx Fallback ({type(exc).__name__}: {exc})")

        markdown = self._parse_with_python_docx(file_path)
        return ParseOutput(markdown, "python-docx", {}, warnings)

    # --------------------------------------------------------------- docling
    @staticmethod
    def _check_docling_installed() -> bool:
        # import 대신 find_spec: docling은 import만으로 torch 등을 로드해 수 초가 걸립니다.
        if importlib.util.find_spec("docling") is None:
            return False
        return True

    def _parse_with_docling(self, file_path: Path) -> str:
        with self._docling_lock:
            if DocxParser._docling_converter is None:
                from docling.document_converter import DocumentConverter

                DocxParser._docling_converter = DocumentConverter()
            converter = DocxParser._docling_converter

        result = converter.convert(str(file_path))
        status = str(getattr(result, "status", "")).lower()
        if "failure" in status:
            raise ParserError(f"Docling 변환 실패 (status={status})", source=file_path)
        return result.document.export_to_markdown()

    # ---------------------------------------------------------- python-docx
    def _parse_with_python_docx(self, file_path: Path) -> str:
        try:
            import docx
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise ParserError("python-docx가 설치되어 있지 않습니다", source=file_path) from exc

        try:
            document = docx.Document(str(file_path))
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise ParserError(f"Word 파일을 열 수 없습니다(손상 가능성): {exc}", source=file_path) from exc
        except Exception as exc:  # docx.opc.exceptions.PackageNotFoundError 등
            raise ParserError(f"Word 파일을 열 수 없습니다: {exc}", source=file_path) from exc

        blocks: List[str] = []
        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                text = self._paragraph_to_markdown(Paragraph(child, document), qn)
                if text:
                    blocks.append(text)
            elif child.tag == qn("w:tbl"):
                text = self._table_to_markdown(Table(child, document))
                if text:
                    blocks.append(text)
        return "\n\n".join(blocks)

    def _paragraph_to_markdown(self, paragraph: Any, qn: Any) -> str:
        text = re.sub(r"\s+", " ", paragraph.text or "").strip()
        if not text:
            return ""

        level = self._heading_level(paragraph, qn)
        if level is not None:
            # 헤더 기반 청킹이 동작하려면 Word 제목 스타일을 반드시 '#'으로 바꿔야 합니다.
            return f"{'#' * min(level, 6)} {text}"

        style_name = (paragraph.style.name or "") if paragraph.style is not None else ""
        p_pr = paragraph._p.pPr
        is_list = "list" in style_name.lower() or (p_pr is not None and p_pr.numPr is not None)
        return f"- {text}" if is_list else text

    @staticmethod
    def _heading_level(paragraph: Any, qn: Any) -> Optional[int]:
        style = paragraph.style
        name = (style.name or "").strip() if style is not None else ""
        if name.lower() in ("title", "제목"):
            return 1
        m = _HEADING_STYLE_RE.match(name)
        if m:
            return int(m.group(1))
        # 사용자 정의 스타일("회사_장제목" 등)도 개요 수준(outlineLvl)이 설정되어 있으면 제목입니다.
        # 이를 놓치면 사내 템플릿 문서가 통째로 "헤더 없는 문서"가 되어 섹션 인용이 불가능해집니다.
        p_pr = paragraph._p.pPr
        if p_pr is not None:
            outline = p_pr.find(qn("w:outlineLvl"))
            if outline is not None:
                try:
                    lvl = int(outline.get(qn("w:val")))
                except (TypeError, ValueError):
                    return None
                if 0 <= lvl <= 8:  # 9는 "본문 수준"
                    return lvl + 1
        return None

    @staticmethod
    def _table_to_markdown(table: Any) -> str:
        rows: List[List[str]] = []
        for row in table.rows:
            # 가로 병합 셀은 python-docx가 같은 셀을 반복 반환합니다. Excel 파서의
            # forward-fill과 같은 효과이므로 그대로 둡니다(각 열이 자기 값을 가짐).
            rows.append([re.sub(r"\s+", " ", cell.text or "").strip() for cell in row.cells])
        rows = [r for r in rows if any(r)]
        if not rows:
            return ""

        width = max(len(r) for r in rows)
        if width == 1:
            # 1열 표는 대부분 "글상자처럼 쓴 표"(레이아웃용)이므로 문단으로 풀어 줍니다.
            return "\n\n".join(r[0] for r in rows if r[0])
        if len(rows) == 1:
            return " | ".join(c for c in rows[0] if c)
        return build_markdown_table(rows[0], rows[1:])
