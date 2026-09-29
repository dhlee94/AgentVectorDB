"""BM25용 한국어 토크나이저 (Kiwi 형태소 분석 + 코드/숫자 원형 보존).

[왜 필요한가]
BM25는 "토큰이 정확히 일치하는가"로 점수를 매깁니다. 공백 분리를 쓰면
"견적서의", "견적서를", "견적서는"이 전부 다른 토큰이라 질문 "견적서"와 하나도 매칭되지 않습니다.
Kiwi로 조사/어미를 떼고 명사·어근만 남겨야 한국어 키워드 검색이 실제로 동작합니다.

[형태소 분석만으로 부족한 점]
형태소 분석기는 "SKU-00123", "A4", "v2.1" 같은 코드를 "SKU", "-", "00123"으로 쪼갭니다.
하이브리드 검색에서 BM25를 쓰는 가장 큰 이유가 바로 이런 고유 코드를 정확히 잡는 것이므로,
영숫자 코드는 정규식으로 "원형 그대로"도 한 번 더 토큰에 넣습니다.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

# 검색에 의미 있는 품사만 남깁니다.
#   NNG/NNP/NNB/NR/NP: 명사류, SL: 외국어, SN: 숫자, SH: 한자, XR: 어근,
#   VV/VA: 동사/형용사 어간 ("할인하다" → "할인"은 NNG, "비싸다" → "비싸" VA)
_KEEP_TAG_PREFIXES = ("NN", "NR", "NP", "SL", "SN", "SH", "XR", "VV", "VA")

# 2글자 이상, 영문/숫자가 섞인 코드 (하이픈·점·슬래시·밑줄 허용): SKU-00123, A4, v2.1, 2025-07-15
_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_./]*[A-Za-z0-9]")
# Kiwi 미설치 시 대체 토크나이저: 한글 덩어리 / 영숫자 덩어리
_FALLBACK_RE = re.compile(r"[가-힣]+|[A-Za-z0-9]+")


class KoreanTokenizer:
    """`tokenizer(text) -> List[str]`. 인덱싱과 질의에 반드시 같은 인스턴스(설정)를 써야 합니다."""

    def __init__(self, use_kiwi: bool = True) -> None:
        self._kiwi: Optional[Any] = None
        # Kiwi 객체는 스레드 안전성이 보장되지 않아 락으로 보호합니다 (웹 서버 멀티스레드 대비).
        self._lock = threading.Lock()
        if use_kiwi:
            try:
                from kiwipiepy import Kiwi

                self._kiwi = Kiwi()
            except ImportError:
                logger.warning(
                    "kiwipiepy 미설치 → 정규식 토크나이저로 대체합니다. "
                    "조사가 붙은 단어가 매칭되지 않아 BM25 품질이 크게 떨어집니다."
                )

    @property
    def uses_kiwi(self) -> bool:
        return self._kiwi is not None

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        text = unicodedata.normalize("NFC", text or "")
        if not text.strip():
            return []
        if self._kiwi is None:
            tokens = [t.lower() for t in _FALLBACK_RE.findall(text)]
        else:
            with self._lock:
                morphs = self._kiwi.tokenize(text)
            tokens = [m.form.lower() for m in morphs if m.tag.startswith(_KEEP_TAG_PREFIXES)]
        tokens.extend(code.lower() for code in _CODE_RE.findall(text))
        return tokens

    def tokenize_many(self, texts: Iterable[str]) -> List[List[str]]:
        """인덱싱용 일괄 토큰화. (수만 건 재구축 시 호출 오버헤드를 줄이기 위해 분리)"""
        return [self.tokenize(t) for t in texts]
