# 엔터프라이즈급 비정형 문서 RAG 파이프라인 아키텍처 및 구현 명세서

## 1. 프로젝트 개요
본 프로젝트는 로컬 환경에 저장된 복잡한 비정형 문서(다단 PDF, 병합된 표가 있는 Excel, Word)를 자동으로 파싱하고, 문맥 유실 없이 정확하게 정보를 검색하여 출처 기반의 답변을 생성하는 **엔터프라이즈급 RAG(Retrieval-Augmented Generation) 시스템**입니다.

일반적인 튜토리얼 수준을 넘어, 실무 환경에서 발생하는 표(Table) 붕괴, 문맥 단절, 한국어 검색 형태소 문제, 맥락 없는 인용 등의 엣지 케이스(Edge Case)를 완벽하게 방어하도록 설계되었습니다.

---

## 2. 전체 시스템 아키텍처 (Data Flow)

1. **Ingestion (문서 적재):** 해시(SHA-256) 기반 매니페스트를 통해 신규 및 변경된 파일만 증분 업데이트(Incremental Update).
2. **Routing & Parsing:** 파일 포맷과 문서 상태에 따라 최적의 파서(Docling, PyMuPDF, Pandas 등)로 자동 라우팅하여 Markdown으로 변환.
3. **Context-Aware Chunking:** 헤더 기반의 논리적 섹션 분할(Parent) 후, 고정밀 검색을 위한 조각(Child) 분할 수행. 모든 조각에 표 헤더와 메타데이터 강제 주입.
4. **Indexing:** Child 청크는 Vector DB(Chroma)와 Sparse 인덱스(BM25)에 동시 저장. 원본 Parent 청크는 디스크(LocalFileStore)에 영속 저장.
5. **Hybrid Retrieval:** 질문 인입 시 Dense와 Sparse 검색 결과를 RRF로 융합하고, 점수가 가장 높은 Child의 Parent를 최종 반환 (Small-to-Big).
6. **Generation:** 출처(파일명, 섹션, 페이지)를 명시하도록 강제된 프롬프트를 통해 LLM(Claude)이 신뢰할 수 있는 답변 생성.

---

## 3. 4대 핵심 구현 전략

### 3.1. 무손실 문서 파싱 (Document Parsing)
모든 비정형 문서를 공통 중간 표현(IR)인 **Markdown**으로 변환합니다.
* **복잡한 PDF / Word:** `Docling`을 최우선으로 사용하여 다단 레이아웃과 표 구조(Table)를 Markdown 형식으로 완벽히 복원.
* **대용량 / 단순 PDF:** 대용량 PDF는 처음부터, 그 외에는 Docling 실패·타임아웃·본문 누락 시 고속 추출기인 `PyMuPDF4LLM`으로 Fallback(우회) 처리. (비밀번호가 걸린 PDF는 건너뛰고 실패 목록에 기록)
* **Excel:** 단순 표는 `pandas.to_markdown()`으로 처리하되, 병합 셀이나 다단 헤더가 있는 복잡한 시트는 LLM이 읽기 쉽도록 **행 단위 서술형 텍스트 템플릿**(예: `[행5] 구분: 사무용품, 가격_단가: 50`, 시트명은 섹션 태그에 포함)으로 변환.

### 3.2. 문맥 보존 청킹 (Context-Aware Chunking)
정보의 경계를 기계적인 글자 수가 아닌 **의미 단위**로 나눕니다. (Small-to-Big 전략)
* **Parent 청크 (생성용):** `MarkdownHeaderTextSplitter`를 사용해 `#`, `##` 등 마크다운 헤더(장/절)를 기준으로 문서를 분할.
* **Child 청크 (검색용):** Parent를 300~500자 단위로 잘게 쪼개어 검색 정밀도 확보.
* **표 파편화 방지:** 커스텀 `TableAwareSplitter`를 구현하여 표 중간 단절을 막고, 두 번째 조각부터는 **원본 표의 헤더를 최상단에 강제 복사**하여 데이터 의미 유실 방지.
* **메타데이터 태그 주입:** 검색 벡터에 출처 정보가 반영되도록 모든 청크 본문 맨 앞에 `[문서명: 2024_견적서.pdf, 섹션: 2장 > 2.1 단가]` 텍스트를 강제로 박아 넣음.

### 3.3. 하이브리드 검색 (Hybrid Retrieval)
의미 기반 검색(Dense)과 키워드 검색(Sparse)을 결합하여 고유명사와 수치 검색의 한계를 극복합니다.
* **RRF(Reciprocal Rank Fusion) 융합:** Child 단위에서 Vector 검색과 BM25 검색을 수행한 뒤 순위 기반으로 점수를 병합.
* **Parent 점수 최댓값(Max) 반영:** 합산(Sum) 방식을 쓸 경우 쪼개진 표 조각이 많은 엉뚱한 섹션이 상위를 독식하는 문제를 방지하기 위해, **소속 Child 중 최고 점수**만 Parent의 점수로 대표하도록 설계.
* **한국어 최적화 토크나이저:** `kiwipiepy` 형태소 분석기를 도입하여 한국어 조사 분리 및 "SKU-00123" 같은 고유 제품 코드의 원형을 보존하여 인덱싱.

### 3.4. 출처 표기 기반 답변 생성 (Citation Generation)
LLM의 환각(Hallucination)을 원천 차단합니다.
* 프롬프트에 제공된 컨텍스트(Parent 청크) 번호 `[n]`에 기반해서만 답변하도록 시스템 프롬프트 강제.
* 문서에 근거가 없을 경우 "제공된 문서에서 확인할 수 없습니다"로 답변 거절.

---

## 4. 실무 엣지 케이스(Edge Case) 방어 로직

과제 및 튜토리얼 수준에서는 알 수 없는 실제 운영 환경의 문제들을 다음과 같이 해결했습니다.

1. **LangChain 줄바꿈 버그 해결:** `MarkdownHeaderTextSplitter`가 빈 줄을 합쳐버려 문단 가독성을 해치는 현상을 후처리로 복원.
2. **캡션 고립(Orphan) 방지:** 표의 "단위: 원" 같은 캡션이나 제목만 덩그러니 별도 청크로 떨어져 나가는 현상을 막아 다음 블록에 강제 병합.
3. **macOS 한글 파일명 정규화 (NFC):** Mac에서 생성된 자모 분리형(NFD) 파일명을 정규화하여 검색어와의 불일치 문제 해결.
4. **Excel 포맷팅 보존:** 제품 코드의 앞자리 `0` (예: `00123`)이 지워지거나, 퍼센트 값(`15%`)이 실수(`0.15`)로 변환되어 검색을 방해하는 현상 방어.
5. **Chroma DB 에러 방지:** 메타데이터에 `None` 값이 들어갈 경우 DB 저장이 터지는 현상을 막기 위해 예외 딕셔너리 필터링 적용.
6. **페이지 마커 보존:** 섹션이 페이지 중간에서 시작해도 원래 페이지 번호를 알 수 있도록 헤더 분할 전 페이지 마커를 삽입.

---

## 5. 기술 스택 (Tech Stack)

* **Orchestration:** LangChain (`langchain`, `langchain-core`, `langchain-text-splitters`)
* **Parsing:** Docling, PyMuPDF (`pymupdf4llm`), Pandas, Openpyxl, python-docx
* **Embedding & LLM:** BAAI/bge-m3 (Local Embedding), Claude Haiku 4.5 (`claude-haiku-4-5`, 공식 `anthropic` SDK)
* **Vector DB & Docstore:** ChromaDB (Local), `LocalFileStore` (Persistent Docstore)
* **Sparse Search:** `rank-bm25`, `kiwipiepy` (한국어 형태소 분석)

---

## 6. 디렉토리 구조

```text
config.yaml                   # 전체 설정 파일 (파서, 청커, 검색, 모델 등 — 여기서만 수정)
rag/
├── config.py                 # 설정 스키마(기본값) 및 config.yaml 로더·검증
├── exceptions.py             # 커스텀 예외 처리
├── schemas.py                # dataclass 기반 입출력 스키마 정의
├── markdown_utils.py         # Markdown 줄바꿈 복원 및 포맷팅 유틸
├── parsers/                  # 파서 모듈 (PDF, Excel, Word 라우팅 및 처리)
│   ├── base.py
│   ├── pdf_parser.py         # Docling -> PyMuPDF Fallback
│   ├── excel_parser.py       # to_markdown 및 서술형 템플릿 변환
│   ├── docx_parser.py
│   └── router.py             # 파일 확장자 및 상태 기반 자동 라우터
├── chunking/                 # 청킹 모듈
│   ├── table_splitter.py     # 표 헤더 복사 및 단절 방지 스플리터
│   └── chunker.py            # Parent-Child 대소 분할기
├── indexing/                 # 인덱싱 모듈
│   ├── korean_tokenizer.py   # Kiwi 한국어 형태소 및 원형 보존 토크나이저
│   └── store.py              # Chroma + Docstore + BM25 + 해시 매니페스트 관리
├── retrieval/                # 검색 모듈
│   └── hybrid.py             # Dense + Sparse RRF 결합 기반 하이브리드 리트리버
├── generation/               # 답변 생성 모듈
│   └── answerer.py           # Citation 프롬프트 및 응답 검증
├── pipeline.py               # 파이프라인 통합 (Ingest & Query)
└── __main__.py               # CLI 엔트리포인트 (python -m rag)

---

## 7. 실행 방법

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) API 키 설정 — .env는 git에 커밋되지 않습니다
cp .env.example .env
#    .env를 열어 ANTHROPIC_API_KEY=sk-ant-... 에 실제 키 입력

# 3) 설정 확인/수정 (모델, 청크 크기, 검색 가중치 등) — config.yaml

# 4) 문서 인덱싱 → 질문
python -m rag ingest ./data/inbox
python -m rag query "볼펜 할인율은?"
```

### 테스트

```bash
pip install -r requirements-dev.txt
python -m pytest        # 84개, 약 15초
```
실제 Claude API와 bge-m3 모델 없이 실행됩니다 (가짜 LLM 클라이언트와 가짜 임베딩 사용, 샘플 문서는 테스트 시작 시 생성). 비용이나 네트워크가 필요 없습니다.
