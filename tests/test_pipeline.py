"""파이프라인 통합: 인제스트·변경 감지, 하이브리드 검색, 인용 답변, 오류 처리, 임베딩 자기검사."""

from __future__ import annotations

from typing import List

import pytest

from rag.exceptions import IndexConfigMismatchError, IndexingError
from rag.generation import NO_ANSWER
from tests.conftest import make_pipeline


# ------------------------------------------------------------- ingest
def test_ingest_report_and_unchanged_skip(indexed, inbox):
    report = indexed.ingest(str(inbox))
    assert len(report.unchanged) == 3 and not report.indexed
    assert [f.error_type for f in report.failures] == ["EncryptedDocumentError"]


def test_change_detection_and_removal(indexed, inbox):
    import docx

    path = inbox / "규정_테스트.docx"
    d = docx.Document(str(path))
    d.add_paragraph("신규 조항: 재택근무는 주 2회까지 허용된다.")
    d.save(str(path))
    (inbox / "manual.pdf").unlink()

    report = indexed.ingest(str(inbox))
    assert [p.split("/")[-1] for p in report.indexed] == ["규정_테스트.docx"]
    assert [p.split("/")[-1] for p in report.removed] == ["manual.pdf"]
    assert "재택근무" in indexed.retriever.retrieve("재택근무 몇 회")[0].document.page_content
    assert indexed.retriever.retrieve("단가", where={"source": "manual.pdf"}) == []


def test_index_persists_across_instances(indexed, tmp_path, fake_llm):
    again = make_pipeline(tmp_path / "store", fake_llm)
    assert "볼펜" in again.retriever.retrieve("볼펜 할인율")[0].document.page_content


def test_embedding_model_mismatch_guard(indexed, tmp_path, fake_llm):
    with pytest.raises(IndexConfigMismatchError):
        make_pipeline(tmp_path / "store", fake_llm, embedding_model="other-model")


# ---------------------------------------------------------- retrieval
def test_bm25_finds_korean_with_particle_and_codes(indexed):
    # 가짜 임베딩(무작위 벡터)이라 Dense는 무의미 → BM25가 정확히 찾아야 함
    top = indexed.retriever.retrieve("볼펜의 할인율은?")[0]
    assert "볼펜" in top.document.page_content and "sparse" in top.retrievers
    assert "부품-027" in indexed.retriever.retrieve("부품-027 가격")[0].document.page_content


def test_metadata_filter(indexed):
    results = indexed.retriever.retrieve("단가", where={"source": "manual.pdf"})
    assert results and all(r.document.metadata["source"] == "manual.pdf" for r in results)


def test_langchain_retriever_interface(indexed):
    docs = indexed.retriever.invoke("볼펜 할인율")
    assert docs and "retrieval_score" in docs[0].metadata


def test_korean_tokenizer_keeps_codes():
    from rag.indexing import KoreanTokenizer

    tokens = KoreanTokenizer()("견적서의 SKU-00123 단가는 15%입니다")
    assert "견적서" in tokens and "단가" in tokens  # 조사 분리
    assert "sku-00123" in tokens  # 코드 원형 보존


# --------------------------------------------------------- generation
def test_answer_with_citation_validation(indexed, fake_llm):
    fake_llm.reply = "볼펜의 할인율은 10%입니다 [1]. 사무용품 구분입니다[1, 9]. 추가 정보 [7]."
    answer = indexed.query("볼펜의 할인율은?")
    assert "[9]" not in answer.answer and "[7]" not in answer.answer
    assert answer.grounded and answer.citations[0].source == "견적_테스트.xlsx"
    assert any("[7, 9]" in w for w in answer.warnings)


def test_request_parameters_for_haiku(indexed, fake_llm):
    indexed.query("볼펜의 할인율은?")
    kwargs = fake_llm.last
    assert kwargs["model"] == "claude-haiku-4-5" and kwargs["temperature"] == 0.0
    assert not any(k in kwargs for k in ("fallbacks", "betas", "output_config", "thinking"))
    user_message = kwargs["messages"][0]["content"]
    assert user_message.startswith("<documents>") and user_message.rstrip().endswith("질문: 볼펜의 할인율은?")
    assert "[문서명:" not in user_message  # 태그 줄은 XML 속성과 중복이라 제거


def test_no_answer_is_not_grounded(indexed, fake_llm):
    fake_llm.reply = NO_ANSWER
    answer = indexed.query("회사 창립일은?")
    assert not answer.grounded and answer.citations == []


def test_uncited_answer_is_flagged(indexed, fake_llm):
    fake_llm.reply = "그냥 답변입니다."
    answer = indexed.query("볼펜")
    assert not answer.grounded and any("인용 번호가 없는" in w for w in answer.warnings)


def test_empty_index_skips_llm(tmp_path, fake_llm):
    pipe = make_pipeline(tmp_path / "empty_store", fake_llm)
    answer = pipe.query("아무 질문")
    assert answer.answer == NO_ANSWER and fake_llm.calls == []  # 근거 0건이면 LLM 호출 안 함


def test_rate_limit_returns_sources(indexed, fake_llm):
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    fake_llm.raise_exc = anthropic.RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )
    answer = indexed.query("볼펜의 할인율은?")
    assert "답변 생성에 실패" in answer.answer and "다시 시도" in answer.answer
    assert answer.citations and not answer.grounded


# ------------------------------------------------ embedding self-test
class _InconsistentEmbeddings:
    """단건 인코딩만 틀리는 장치(MPS 버그) 흉내: embed_query가 엉뚱한 벡터를 반환."""

    def __init__(self, device: str = "mps", broken: bool = True) -> None:
        self.device, self.broken = device, broken

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, float(len(t)), 0.0] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0, 0.0, 1.0] if self.broken else [1.0, float(len(text)), 0.0]


def test_embedding_self_test_detects_mismatch():
    from rag.indexing.store import embedding_self_test

    assert embedding_self_test(_InconsistentEmbeddings(broken=True)) < 0.5
    assert embedding_self_test(_InconsistentEmbeddings(broken=False)) > 0.999


def _patch_hf(monkeypatch, cpu_broken: bool):
    """HuggingFaceEmbeddings를 가짜로 바꿔 MPS 실패 → CPU 전환 경로를 모델 다운로드 없이 검증."""
    import sys
    import types

    loaded: List[str] = []

    class FakeHF(_InconsistentEmbeddings):
        def __init__(self, model_name, model_kwargs, encode_kwargs):
            device = model_kwargs.get("device", "mps")
            loaded.append(device)
            super().__init__(device=device, broken=(device != "cpu") or cpu_broken)
            self._client = types.SimpleNamespace(device=device)

    monkeypatch.setitem(sys.modules, "langchain_huggingface", types.SimpleNamespace(HuggingFaceEmbeddings=FakeHF))
    return loaded


def test_build_embeddings_falls_back_to_cpu(monkeypatch):
    from rag.config import StoreConfig
    from rag.indexing.store import build_default_embeddings

    loaded = _patch_hf(monkeypatch, cpu_broken=False)
    embeddings = build_default_embeddings(StoreConfig())
    assert loaded == ["mps", "cpu"] and embeddings.device == "cpu"


def test_build_embeddings_fails_if_cpu_also_broken(monkeypatch):
    from rag.config import StoreConfig
    from rag.indexing.store import build_default_embeddings

    _patch_hf(monkeypatch, cpu_broken=True)
    with pytest.raises(IndexingError, match="CPU 전환 후에도"):
        build_default_embeddings(StoreConfig())


# -------------------------------------------------- citation post-processing
@pytest.mark.parametrize(
    "raw, expected",
    [
        # 실제 발생한 버그: 전역 공백 정리가 ".cursorrules" 앞 공백을 지웠음
        ("AGENTS.md, CLAUDE.md, .cursorrules의 차이 [1].", "AGENTS.md, CLAUDE.md, .cursorrules의 차이 [1]."),
        ("## 3. .cursorrules — Cursor 표준 [1]", "## 3. .cursorrules — Cursor 표준 [1]"),
        ("값은 1 , 2 입니다 [1].", "값은 1 , 2 입니다 [1]."),  # 인용과 무관한 공백은 그대로
        # 무효 인용을 지운 자리의 공백만 정리
        ("추가 정보 [7].", "추가 정보."),
        ("근거 [1, 9]. 그리고 [8], 끝", "근거 [1]. 그리고, 끝"),
    ],
)
def test_citation_cleanup_only_touches_removed_citations(raw, expected):
    from rag.generation.answerer import AnswerGenerator

    assert AnswerGenerator._validate_citations(raw, n_contexts=3)[0] == expected


def test_citations_listed_in_index_order(indexed, fake_llm):
    fake_llm.reply = "첫 사실 [3]. 둘째 사실 [1]."
    answer = indexed.query("볼펜 할인율과 부품 가격")
    assert [c.index for c in answer.citations] == [1, 3]
