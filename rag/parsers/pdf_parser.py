"""PDF 파서: 사전 검사 → Docling(정확도) → PyMuPDF4LLM(속도/Fallback).

[설계 의도]
- 전략 A의 핵심은 "표·다단·수식이 있는 PDF를 무손실로 Markdown화"입니다. 이 품질은
  Docling(레이아웃 모델 + TableFormer)이 가장 좋지만, 느리고 무겁고 가끔 실패합니다.
- 그래서 세 겹의 방어선을 둡니다.
  1) 사전 검사(PyMuPDF, 수십 ms): 암호화/페이지 수/스캔 여부를 먼저 판단해
     애초에 Docling에 보내면 안 되는 문서를 거릅니다.
  2) Docling 실행 후 "품질 검증": 예외가 안 났어도 본문 대부분이 누락되는 경우가 있어
     PyMuPDF 원시 텍스트량과 비교합니다. 조용한 실패가 가장 위험하기 때문입니다.
  3) Fallback: 어떤 이유로든 Docling이 실패하면 PyMuPDF4LLM으로 재파싱합니다.
- 페이지 경계마다 `<!-- page: N -->` 마커를 넣어 청커가 인용용 페이지 번호를 추적할 수 있게 합니다.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import fitz  # PyMuPDF

from rag.config import ParserConfig
from rag.exceptions import EmptyDocumentError, EncryptedDocumentError, ParserError
from rag.markdown_utils import normalize_text, page_marker, strip_page_markers
from rag.parsers.base import BaseParser, ParseOutput

logger = logging.getLogger(__name__)


@dataclass
class PdfProfile:
    """사전 검사 결과."""

    page_count: int
    size_mb: float
    sampled_pages: int
    sampled_text_chars: int
    raw_text_chars: int  # 전체 페이지 원시 텍스트 글자 수 (Docling 품질 검증 기준값)

    @property
    def avg_chars_per_page(self) -> float:
        return self.sampled_text_chars / self.sampled_pages if self.sampled_pages else 0.0


class PDFParser(BaseParser):
    supported_extensions = frozenset({".pdf"})

    # Docling 컨버터는 레이아웃/표 모델을 메모리에 올리는 데 수 초~수십 초가 걸립니다.
    # 파일마다 새로 만들면 인제스트 시간이 폭증하므로 클래스 수준에서 캐시합니다.
    # (OCR on/off 두 가지 설정만 존재하므로 dict 키는 bool)
    _docling_converters: dict = {}
    _docling_lock = threading.Lock()

    def __init__(self, config: ParserConfig | None = None) -> None:
        super().__init__(config)
        self._docling_available = self.config.use_docling and self._check_docling_installed()

    # ------------------------------------------------------------------ main
    def _parse(self, file_path: Path) -> ParseOutput:
        profile = self._preflight(file_path)
        warnings: List[str] = []
        metadata: dict = {
            "page_count": profile.page_count,
            "size_mb": round(profile.size_mb, 2),
        }

        is_large = (
            profile.page_count > self.config.large_pdf_page_threshold
            or profile.size_mb > self.config.large_pdf_size_mb
        )
        is_scanned = profile.avg_chars_per_page < self.config.scanned_chars_per_page
        metadata["is_scanned_suspected"] = is_scanned

        # ---- 라우팅 결정 --------------------------------------------------
        if is_scanned and not self._docling_available:
            # 텍스트 레이어가 없으면 PyMuPDF는 빈 문자열만 돌려줍니다. OCR 수단이 없으므로
            # 억지로 진행해 빈 청크를 만들기보다 명확한 사유와 함께 실패시키는 편이 낫습니다.
            raise EmptyDocumentError(
                "텍스트 레이어가 없는 스캔 PDF로 보이며, OCR(Docling)을 사용할 수 없습니다",
                source=file_path,
            )

        if is_large and not is_scanned:
            logger.info("대용량 PDF(%d쪽) → PyMuPDF4LLM 고속 경로", profile.page_count)
            markdown = self._parse_with_pymupdf(file_path)
            metadata["route_reason"] = "large_pdf"
            return ParseOutput(markdown, "pymupdf4llm", metadata, warnings)

        if not self._docling_available:
            metadata["route_reason"] = "docling_unavailable"
            markdown = self._parse_with_pymupdf(file_path)
            return ParseOutput(markdown, "pymupdf4llm", metadata, warnings)

        # ---- Docling 우선 시도 ---------------------------------------------
        try:
            markdown = self._parse_with_docling(file_path, ocr=is_scanned)
            self._verify_docling_quality(markdown, profile, is_scanned, file_path)
            metadata["route_reason"] = "scanned_ocr" if is_scanned else "complex_layout"
            return ParseOutput(markdown, "docling", metadata, warnings)
        except EncryptedDocumentError:
            raise
        except Exception as exc:  # noqa: BLE001 — Docling 내부 예외 종류가 매우 다양함
            if is_scanned:
                # 스캔 PDF는 PyMuPDF로 가도 텍스트가 없으므로 Fallback이 의미가 없습니다.
                raise ParserError(f"스캔 PDF OCR 실패: {exc}", source=file_path) from exc
            warnings.append(f"Docling 실패 → PyMuPDF4LLM Fallback ({type(exc).__name__}: {exc})")

        markdown = self._parse_with_pymupdf(file_path)
        metadata["route_reason"] = "docling_fallback"
        return ParseOutput(markdown, "pymupdf4llm", metadata, warnings)

    # ------------------------------------------------------------ preflight
    def _preflight(self, file_path: Path) -> PdfProfile:
        """PyMuPDF로 빠르게 문서 성격을 파악합니다 (전체 파싱 대비 1% 미만의 비용)."""
        try:
            doc = fitz.open(file_path)
        except Exception as exc:  # fitz.FileDataError 등 — 손상 파일
            raise ParserError(f"PDF를 열 수 없습니다(손상 가능성): {exc}", source=file_path) from exc

        try:
            # is_encrypted여도 needs_pass가 False면 "소유자 암호"만 걸린 문서(인쇄/복사 제한)로
            # 텍스트 추출은 가능합니다. 둘을 구분하지 않으면 멀쩡한 문서를 대량으로 놓칩니다.
            if doc.needs_pass and not doc.authenticate(""):
                raise EncryptedDocumentError("비밀번호가 필요한 PDF입니다", source=file_path)

            page_count = doc.page_count
            if page_count == 0:
                raise EmptyDocumentError("페이지가 0개인 PDF입니다", source=file_path)

            # 앞쪽 페이지만 보면 표지/목차(텍스트 적음) 때문에 스캔으로 오판할 수 있어
            # 문서 전체에 고르게 퍼진 페이지를 샘플링합니다.
            sample_n = min(self.config.preflight_sample_pages, page_count)
            step = max(page_count // sample_n, 1)
            sample_idx = set(list(range(0, page_count, step))[:sample_n])

            raw_chars = 0
            sampled_chars = 0
            for i in range(page_count):
                n = len(doc.load_page(i).get_text("text").strip())
                raw_chars += n
                if i in sample_idx:
                    sampled_chars += n

            return PdfProfile(
                page_count=page_count,
                size_mb=file_path.stat().st_size / (1024 * 1024),
                sampled_pages=len(sample_idx),
                sampled_text_chars=sampled_chars,
                raw_text_chars=raw_chars,
            )
        finally:
            doc.close()

    # --------------------------------------------------------------- docling
    @staticmethod
    def _check_docling_installed() -> bool:
        # import 대신 find_spec: docling은 import만으로 torch 등을 로드해 수 초가 걸립니다.
        if importlib.util.find_spec("docling") is None:
            logger.warning(
                "docling이 설치되어 있지 않습니다. 모든 PDF를 PyMuPDF4LLM으로 처리합니다. "
                "(표 구조 품질 저하 가능 — `pip install docling` 권장)"
            )
            return False
        return True

    def _get_docling_converter(self, ocr: bool) -> Any:
        with self._docling_lock:
            if ocr in self._docling_converters:
                return self._docling_converters[ocr]

            # 무거운 import는 실제로 필요할 때만 수행 (모듈 import 시간 단축, 미설치 환경 대응)
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = ocr  # 텍스트 레이어가 있는 PDF에 OCR을 켜면 느리고 오히려 오탈자가 생김
            options.do_table_structure = True  # 전략 A의 핵심: 표 구조 복원
            # 표 셀을 PDF 원본 텍스트와 매칭 → OCR/모델 추정 대신 원문 글자를 그대로 사용
            options.table_structure_options.do_cell_matching = True
            # 버전에 따라 없는 옵션이므로 방어적으로 설정
            if hasattr(options, "document_timeout"):
                options.document_timeout = self.config.docling_timeout_sec

            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
            self._docling_converters[ocr] = converter
            return converter

    def _parse_with_docling(self, file_path: Path, ocr: bool) -> str:
        converter = self._get_docling_converter(ocr)
        result = converter.convert(str(file_path))

        status = str(getattr(result, "status", "")).lower()
        if "failure" in status:
            raise ParserError(f"Docling 변환 실패 (status={status})", source=file_path)
        if "partial" in status:
            # 부분 성공(타임아웃 등)은 일부 페이지가 빠졌다는 뜻 → 품질 검증에서 걸러지도록 로그만 남김
            logger.warning("Docling 부분 성공: %s", file_path.name)

        document = result.document
        page_numbers = sorted(getattr(document, "pages", {}) or {})

        # 페이지별로 export해야 페이지 마커를 넣을 수 있습니다.
        # (문서 전체를 한 번에 export하면 페이지 정보가 사라져 인용에 페이지를 표기할 수 없음)
        if page_numbers:
            try:
                parts: List[str] = []
                for page_no in page_numbers:
                    page_md = document.export_to_markdown(page_no=page_no)
                    parts.append(f"{page_marker(page_no)}\n\n{page_md}")
                return "\n\n".join(parts)
            except TypeError:
                # page_no 인자를 지원하지 않는 구버전 Docling
                logger.debug("Docling 구버전: 페이지 단위 export 미지원 → 전체 export")

        return document.export_to_markdown()

    def _verify_docling_quality(
        self, markdown: str, profile: PdfProfile, is_scanned: bool, file_path: Path
    ) -> None:
        """Docling이 예외 없이 본문을 대량 누락한 "조용한 실패"를 잡아냅니다."""
        if is_scanned:
            # 스캔 PDF는 비교 기준(원시 텍스트)이 없으므로 결과가 비어 있지만 않으면 통과
            if not strip_page_markers(markdown).strip():
                raise ParserError("OCR 결과가 비어 있습니다", source=file_path)
            return

        docling_chars = len(strip_page_markers(normalize_text(markdown)).replace(" ", ""))
        raw_chars = max(profile.raw_text_chars, 1)
        ratio = docling_chars / raw_chars
        if ratio < self.config.min_docling_text_ratio:
            raise ParserError(
                f"Docling 결과가 원문 대비 {ratio:.0%}만 추출됨 (본문 누락 의심)",
                source=file_path,
            )

    # -------------------------------------------------------------- pymupdf
    def _parse_with_pymupdf(self, file_path: Path) -> str:
        """PyMuPDF4LLM으로 페이지별 Markdown을 추출합니다.

        순수 `fitz.get_text()` 대신 pymupdf4llm을 쓰는 이유: 글꼴 크기로 헤더(#)를 추정하고
        표를 Markdown 표로 뽑아 주므로, 다음 단계의 헤더 기반 청킹이 동작할 수 있습니다.
        """
        try:
            import pymupdf4llm
        except ImportError:
            logger.warning("pymupdf4llm 미설치 → fitz 순수 텍스트 추출로 대체 (헤더/표 구조 손실)")
            return self._parse_with_fitz_text(file_path)

        try:
            pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True, show_progress=False)
        except TypeError:
            # show_progress 인자가 없는 구버전
            pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pymupdf4llm 실패(%s) → fitz 순수 텍스트 추출로 대체", exc)
            return self._parse_with_fitz_text(file_path)

        parts: List[str] = []
        # page_chunks=True는 페이지 순서대로 반환됩니다. 메타데이터의 페이지 키 이름이
        # 버전마다 달라("page", "page_number") enumerate로 1-based 번호를 직접 매깁니다.
        for page_no, page in enumerate(pages, start=1):
            text = page.get("text", "") if isinstance(page, dict) else str(page)
            parts.append(f"{page_marker(page_no)}\n\n{text}")
        return "\n\n".join(parts)

    def _parse_with_fitz_text(self, file_path: Path) -> str:
        """최후의 Fallback. 구조는 잃지만 텍스트라도 확보합니다."""
        parts: List[str] = []
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc, start=1):
                parts.append(f"{page_marker(i)}\n\n{page.get_text('text')}")
        return "\n\n".join(parts)
