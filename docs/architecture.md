# RAG 파이프라인 아키텍처 설계도 (Task 1)

> 기준 문서: `rag_requirements.md` — 4대 핵심 전략(A. 무손실 파싱, B. 문맥 보존 청킹, C. 하이브리드 검색, D. 출처 표기)
> 범위: 설계만 다룹니다. 코드 구현(Task 2)은 이 설계를 확정한 뒤 진행합니다.

---

## 0. 설계 원칙

| 원칙 | 설계에 반영한 방식 |
|---|---|
| **검색은 작게, 생성은 크게** | Child(작은 청크)로 검색하고 Parent(섹션 단위)를 LLM에 전달 (Small-to-Big) |
| **정보 손실 없는 변환** | 모든 포맷을 **Markdown**이라는 공통 중간 표현(IR)으로 먼저 변환한 뒤 청킹 |
| **청크는 스스로 설명 가능해야 함** | 청크 텍스트에 `[문서명: …, 섹션: …]`을 박고, 잘린 표에는 헤더를 다시 붙임 |
| **벡터 검색만 믿지 않음** | Dense(의미) + Sparse(BM25, 고유명사·수치)를 RRF로 결합 |
| **근거 없으면 답하지 않음** | 번호가 붙은 컨텍스트와 인용 강제 프롬프트, 근거 부족 시 "확인 불가" 응답 |
| **실패해도 파이프라인은 계속** | 파서 Fallback 체인, 파일 단위 예외 격리, 실패 목록 리포트 |

---

## 1. 전체 데이터 플로우 (한눈에 보기)

```
 ┌────────────────────────────────────────── INGESTION (오프라인 / 배치) ──────────────────────────────────────────┐
 │                                                                                                              │
 │  ./data/inbox/                                                                                               │
 │  (PDF, XLSX, DOCX)                                                                                           │
 │       │                                                                                                      │
 │       ▼                                                                                                      │
 │  [1] DocumentLoader ──── 해시 매니페스트 비교 ──▶ 변경 없음 → SKIP                                             │
 │       │  (신규/변경 파일만)                                                                                    │
 │       ▼                                                                                                      │
 │  [2] ParserRouter ─┬─ PDF(복잡)  ─▶ Docling ──(실패/타임아웃)──▶ PyMuPDF4LLM (Fallback)                       │
 │                    ├─ PDF(대용량)─▶ PyMuPDF4LLM                                                               │
 │                    ├─ XLSX(단순) ─▶ pandas.to_markdown()                                                     │
 │                    ├─ XLSX(병합) ─▶ openpyxl → 행 단위 서술형 템플릿                                           │
 │                    └─ DOCX       ─▶ Docling ──(실패)──▶ python-docx                                           │
 │       │                                                                                                      │
 │       ▼  ParsedDocument { markdown, metadata{source, file_type, pages, parser_used} }                        │
 │  [3] Chunker                                                                                                 │
 │       ├─ 3-1 MarkdownHeaderTextSplitter  (#, ##, ### 기준 → Parent 청크 = 섹션)                               │
 │       ├─ 3-2 Parent 크기 보정           (너무 크면 분할 / 너무 작으면 인접 섹션 병합)                           │
 │       ├─ 3-3 Child 분할                 (RecursiveCharacterTextSplitter, 표는 TableAwareSplitter)              │
 │       ├─ 3-4 표 헤더 재주입              (잘린 표 조각마다 | 항목 | 가격 | + |---| 복사)                        │
 │       └─ 3-5 메타데이터 태그 주입        ("[문서명: XX, 섹션: 장>절]\n" + 본문)                                  │
 │       │                                                                                                      │
 │       ├──────────── Parent 청크 ─────────────┐        ┌──────────── Child 청크 (parent_id 보유) ─────────┐  │
 │       ▼                                      ▼        ▼                                                ▼  │
 │  [4] Docstore (Key-Value)                          Vector DB (Chroma)                     BM25 Index        │
 │      parent_id → Parent Document                   임베딩(bge-m3) + 메타데이터            (Kiwi 형태소 토큰) │
 │      (LocalFileStore, 영속)                         (영속, 디스크)                        (Chroma에서 재구축) │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 ┌────────────────────────────────────────── QUERY (온라인 / 질의 시) ───────────────────────────────────────────┐
 │                                                                                                              │
 │  사용자 질문                                                                                                  │
 │       │                                                                                                      │
 │       ▼                                                                                                      │
 │  [5] HybridRetriever                                                                                         │
 │       ├─ Dense  : Chroma similarity_search(질문, k=20)   ─┐                                                  │
 │       ├─ Sparse : BM25(Kiwi 토큰화된 질문, k=20)          ─┤─▶ RRF 융합 (child 단위)                           │
 │       │                                                   ┘        │                                         │
 │       │                                                            ▼                                         │
 │       │                                   parent_id 기준 중복 제거 → Docstore에서 Parent 조회 (top_n=4~6)       │
 │       │                                                            │                                         │
 │       │                                   (선택) Cross-Encoder 재순위 (bge-reranker-v2-m3)                     │
 │       ▼                                                            ▼                                         │
 │  [6] AnswerGenerator                                                                                         │
 │       ├─ Parent들을 번호 붙인 컨텍스트 블록으로 조립  [1] 파일명 / 섹션 / 페이지 …                                │
 │       ├─ 인용 강제 시스템 프롬프트 + 질문 → LLM (Claude Haiku 4.5, 공식 anthropic SDK)                           │
 │       └─ 출력: 답변 본문 + 인라인 인용 [1][3] + 참고문헌 목록(파일명·섹션·페이지)                                   │
 │                                                                                                              │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 단계별 상세 설계

### [1] 문서 로드 (Document Loading)

- **입력:** 지정 로컬 폴더 (예: `./data/inbox/`), 확장자 화이트리스트 `.pdf`, `.xlsx`, `.xls`, `.docx`
- **처리:**
  - `pathlib.Path.rglob()`으로 재귀 탐색, 숨김 파일 및 `~$` 임시 파일(열려 있는 Office 파일의 잠금 파일)은 제외
  - 파일별 **SHA-256 해시**를 계산하여 `manifest.json`과 비교
    - 신규 → 인덱싱 / 변경 → 기존 청크 삭제 후 재인덱싱 / 삭제된 파일 → 인덱스에서 제거
  - 이렇게 해야 폴더에 파일을 계속 추가하는 운영 환경에서 **매번 전체 재임베딩하는 비용**을 피할 수 있습니다.
- **자동화(선택):** `watchdog`으로 폴더를 감시하여 파일이 들어오면 자동으로 인제스트 트리거
- **예외:** 0바이트 파일, 권한 오류, 암호 걸린 PDF → `skipped` 목록에 사유와 함께 기록하고 다음 파일로 진행

| 라이브러리 | 용도 |
|---|---|
| `pathlib`, `hashlib` (표준) | 파일 탐색, 변경 감지 |
| `watchdog` (선택) | 폴더 실시간 감시 |

---

### [2] 전처리 및 파싱 (Parsing) — 전략 A

모든 파서는 동일한 출력 규격 **`ParsedDocument(markdown: str, metadata: dict)`**를 반환합니다. 후속 단계는 원본 포맷을 몰라도 됩니다.

#### 2-1. PDF 라우팅

```
PDF 입력
  │
  ├─ 사전 검사 (PyMuPDF로 빠르게): 페이지 수, 텍스트 레이어 유무, 파일 크기
  │
  ├─ 페이지 수 > N(예: 300) 또는 크기 > M MB  ──────────────▶ PyMuPDF4LLM (속도 우선)
  │
  └─ 그 외 (표/다단/수식 가능성) ─▶ Docling ─┬─ 성공 ─▶ Markdown
                                           └─ 예외/타임아웃/빈 결과 ─▶ PyMuPDF4LLM (Fallback)
```

- **Docling**: 레이아웃 분석 + TableFormer로 표 구조를 Markdown 표로 복원, 다단 읽기 순서 보정, 스캔 PDF는 OCR 옵션
- **PyMuPDF4LLM**: `fitz` 기반으로 Markdown을 바로 출력(헤더 추정 포함)하는 PyMuPDF 공식 래퍼. 순수 `fitz.get_text()`보다 후속 헤더 분할과 궁합이 좋음
- **Marker**: Docling의 대안. GPU 환경에서 수식(LaTeX) 품질이 중요하면 교체 가능하도록 `BaseParser` 인터페이스로 추상화
- **품질 검증:** 파싱 결과에서 텍스트 길이가 페이지 수 대비 비정상적으로 작으면(스캔 PDF 의심) OCR 모드로 재시도
- 페이지 번호는 Markdown 내에 `<!-- page: 12 -->` 마커로 보존하여 인용 시 사용

#### 2-2. Excel 라우팅

```
XLSX 입력 ─▶ openpyxl로 시트별 로드
  │
  ├─ 병합 셀 없음 & 헤더 1행 ─▶ pandas.read_excel() → df.to_markdown(index=False)
  │
  └─ 병합 셀 존재 or 다단 헤더 ─▶ 1) 병합 범위를 좌상단 값으로 채움(forward-fill)
                                2) 다단 헤더는 "상위헤더_하위헤더"로 평탄화
                                3) 행 단위 서술형 템플릿 변환:
                                   "[시트: 견적, 행 3] 항목: A, 규격: 10mm, 가격: 100"
```

- 시트 하나 = Markdown의 `## 시트명` 섹션 → 이후 헤더 기반 청킹과 자연스럽게 연결
- 빈 행/빈 열 제거, 날짜·통화 서식은 문자열로 정규화 (예: `45123` → `2023-07-15`)

#### 2-3. Word 라우팅

- **Docling**(DOCX 네이티브 지원, 제목 스타일 → `#` 헤더, 표 → Markdown 표)을 기본으로, 실패 시 **python-docx**로 단락/표를 순회하여 Markdown 생성

| 라이브러리 | 용도 |
|---|---|
| `docling` | 복잡 PDF / DOCX → Markdown (1순위) |
| `pymupdf`, `pymupdf4llm` | 대용량 PDF 고속 파싱 및 Fallback, PDF 사전 검사 |
| `marker-pdf` (선택) | Docling 대체 파서 |
| `pandas`, `openpyxl`, `tabulate` | Excel 로드, 병합 셀 탐지, `to_markdown()` (`tabulate` 필수 의존성) |
| `python-docx` | DOCX Fallback |

---

### [3] Parent / Child 청킹 (Chunking) — 전략 B

```
Markdown 전문
   │
   ▼  3-1 MarkdownHeaderTextSplitter(headers=[#, ##, ###], strip_headers=False)
┌──────────── Parent 청크 (섹션 단위, 목표 1,500~3,000자) ────────────┐
│ metadata: parent_id, source, section="2장 > 2.1 가격 정책", pages    │
└──────────────────────────────────────────────────────────────────────┘
   │  3-2 크기 보정: 상한 초과 → 문단 경계로 분할 / 하한 미만 → 같은 상위 헤더의 인접 섹션과 병합
   ▼
   │  3-3 Child 분할 (목표 300~500자, overlap 50자)
   │      ├─ 일반 텍스트: RecursiveCharacterTextSplitter (구분자: "\n\n", "\n", ". ", " ")
   │      └─ 표 블록   : TableAwareSplitter — 표는 행 경계에서만 자름 (행 중간 절단 금지)
   ▼
   │  3-4 표 헤더 재주입: 두 번째 조각부터 원본 표의 헤더 2줄을 앞에 복사
   │      | 항목 | 가격 |
   │      |------|------|
   │      | C    | 300  |   ← 원래는 헤더 없이 잘려나갔던 행
   ▼
   │  3-5 메타데이터 태그 주입 (Child와 Parent 모두)
   │      "[문서명: 2025_견적서.pdf, 섹션: 2장 > 2.1 가격 정책]\n" + 본문
   ▼
Child 청크 { page_content(태그 포함), metadata{ child_id, parent_id, source, section, page, chunk_type } }
```

**설계 포인트**

- **왜 헤더 기반이 1차 분할인가:** 고정 길이 분할은 "2.1 가격 정책"의 문단이 "2.2 환불 규정"과 한 청크에 섞이게 만듭니다. 장/절 경계가 곧 의미 경계이므로 Parent는 섹션을 따릅니다.
- **왜 태그를 텍스트 자체에 넣는가:** 메타데이터 필드는 임베딩되지 않습니다. `"가격은 100원"`만 있는 Child는 어느 문서의 가격인지 알 수 없지만, 태그가 붙으면 임베딩 벡터와 BM25 토큰 모두에 문서명·섹션이 반영되어 **"A 견적서의 가격"** 같은 질의에 걸립니다.
- **왜 표 헤더를 복사하는가:** 헤더 없는 표 조각은 숫자 나열일 뿐이라 검색도, LLM 해석도 불가능합니다.
- **ID 설계:** `parent_id = sha1(source + section_path + 순번)` — 결정적(deterministic) ID여야 재인덱싱 시 upsert/삭제가 가능합니다.
- 서술형 템플릿으로 변환된 Excel 행은 이미 자기 설명적이므로 행 경계 기준으로만 묶습니다.

| 라이브러리 | 용도 |
|---|---|
| `langchain-text-splitters` | `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter` |
| (자체 구현) `TableAwareSplitter` | 표 행 단위 분할 + 헤더 재주입 |
| `tiktoken` 또는 문자 수 | 청크 길이 측정 (한국어는 문자 수 기준이 예측 가능성이 더 높음) |

---

### [4] Vector DB 및 Docstore 인덱싱

```
Child 청크 ──▶ Embedding(bge-m3) ──▶ Chroma collection "children"
                                      (id=child_id, metadata.parent_id 포함)
Child 청크 ──▶ Kiwi 형태소 토큰화 ──▶ BM25 인덱스 (메모리)
Parent 청크 ─────────────────────────▶ Docstore: LocalFileStore(./store/parents) — key=parent_id
매니페스트 ─────────────────────────▶ manifest.json { 파일경로: {hash, parent_ids[], child_ids[]} }
```

- **Vector DB — Chroma:** 로컬 영속(`persist_directory`), 메타데이터 필터(`where={"source": ...}`), 설치가 가벼움. 요구사항의 "로컬 폴더 기반" 운영 형태에 가장 적합합니다.
- **Docstore — `LocalFileStore` + `create_kv_docstore`:** LangChain 기본 예제의 `InMemoryStore`는 프로세스 종료 시 Parent가 사라집니다. 반드시 디스크 영속 저장소를 사용합니다.
- **임베딩 — `BAAI/bge-m3`:** 한국어를 포함한 다국어 성능이 좋고 8K 토큰 입력을 지원하며 로컬에서 실행 가능(문서 외부 유출 없음). API 방식이 필요하면 `EmbeddingProvider` 인터페이스로 교체 가능.
- **BM25 영속화:** `BM25Retriever`는 메모리 전용이므로, 기동 시 Chroma의 `collection.get()`으로 Child 전체를 읽어 재구축합니다(단일 원천 유지). 수십만 청크 이상으로 커지면 아래 확장 경로를 참고하세요.
- **삭제/갱신:** 매니페스트에 기록된 child_ids/parent_ids로 Chroma·Docstore에서 정확히 삭제 후 재삽입

| 라이브러리 | 용도 |
|---|---|
| `chromadb`, `langchain-chroma` | Child 벡터 저장 및 검색 |
| `langchain-huggingface`, `sentence-transformers` | bge-m3 임베딩 로컬 실행 |
| `langchain.storage` (`LocalFileStore`, `create_kv_docstore`) | Parent 영속 저장 |
| `rank-bm25`, `langchain-community` (`BM25Retriever`) | Sparse 인덱스 |
| `kiwipiepy` | 한국어 형태소 분석 기반 BM25 토큰화 |

> **한국어 BM25 주의:** 기본 공백 토큰화는 "견적서의", "견적서를"을 서로 다른 단어로 봅니다. Kiwi로 명사·어근을 추출해야 키워드 매칭이 실제로 작동합니다.

---

### [5] 하이브리드 검색 (Hybrid Search) — 전략 C

```
질문 ─┬─▶ Dense : Chroma(bge-m3)            → child 후보 20개 (순위 r_d)
      └─▶ Sparse: BM25(Kiwi 토큰)            → child 후보 20개 (순위 r_s)
                          │
                          ▼
     RRF 점수(child) = w_d / (60 + r_d) + w_s / (60 + r_s)     (기본 w_d = w_s = 0.5)
                          │
                          ▼
     parent_id별 최고 점수로 집계 → 상위 Parent 4~6개 선택 → Docstore.mget(parent_ids)
                          │
                          ▼ (선택)
     Cross-Encoder 재순위 (bge-reranker-v2-m3) — 질문과 Parent 쌍 직접 비교
```

- **왜 child 단계에서 융합하는가:** LangChain의 `ParentDocumentRetriever`는 Dense 검색만 지원합니다. `EnsembleRetriever`로 Parent 결과끼리 합치면 BM25 쪽도 Parent를 대상으로 해야 해서 Small-to-Big의 이점이 사라집니다. 따라서 **Dense·Sparse 모두 Child를 검색 → RRF로 융합 → Parent로 승격**하는 `HybridParentRetriever`를 자체 구현합니다(`BaseRetriever` 상속).
- **왜 RRF인가:** 코사인 유사도와 BM25 점수는 스케일이 달라 단순 가중합이 불안정합니다. 순위 기반 RRF는 정규화가 필요 없습니다.
- **가중치 튜닝:** 제품 코드·수치 질의가 많으면 Sparse 가중치를 올립니다. 평가셋으로 조정합니다.
- **메타데이터 필터:** 질문에 특정 파일이 지정되면 Chroma `where` 필터와 BM25 후보 필터를 동일하게 적용

| 라이브러리 | 용도 |
|---|---|
| `langchain-core` (`BaseRetriever`) | `HybridParentRetriever` 자체 구현 |
| `langchain` (`EnsembleRetriever`) | RRF 구현 참고 / 단순 모드 대안 |
| `sentence-transformers` (`CrossEncoder`) (선택) | 재순위 |

---

### [6] LLM 답변 생성 (Citation) — 전략 D

**컨텍스트 조립 형식**

```
[1] 파일: 2025_견적서.pdf | 섹션: 2장 > 2.1 가격 정책 | 페이지: 12-13
<Parent 본문>

[2] 파일: 제품목록.xlsx | 섹션: 시트 "단가표" | 행: 3-40
<Parent 본문>
```

**시스템 프롬프트 핵심 규칙**

1. 오직 제공된 컨텍스트 `[n]`에 근거해서만 답한다.
2. 모든 사실 문장 끝에 근거 번호 `[n]`을 붙인다.
3. 컨텍스트에 근거가 없으면 추측하지 말고 "제공된 문서에서 확인할 수 없습니다"라고 답한다.
4. 답변 마지막에 `참고 문서` 목록(파일명 · 섹션 · 페이지)을 출력한다.

**후처리 검증**

- 답변에 등장한 `[n]` 번호가 실제 컨텍스트 범위 안인지 검사 → 범위 밖 인용은 제거하고 경고 로그
- 인용이 하나도 없는 답변은 "근거 부족"으로 표시
- 최종 반환 객체: `RAGAnswer { answer, citations: [{source, section, pages}], retrieved_parents }`

| 라이브러리 | 용도 |
|---|---|
| `anthropic` (공식 SDK) | LLM 호출 — 기본 모델 `claude-haiku-4-5`(temperature 0). 품질이 더 필요하면 `config.yaml`에서 `claude-sonnet-5-5` / `claude-opus-5-5`로 교체 |
| `langchain-core` (`BaseRetriever`) | Retriever를 LCEL 체인과 호환되게 구성 |
| `dataclasses` (표준) | `RAGAnswer`, `Citation` 등 입출력 스키마 |

> 모델명·effort·temperature·재시도 횟수 등 모든 설정은 루트의 `config.yaml`에서 바꿉니다. 모델과 맞지 않는 파라미터 조합은 설정 로드 시 오류로 알려 줍니다. 참고 문서 목록은 LLM이 아니라 코드가 메타데이터로 생성합니다(목록 환각 방지).

---

## 3. 라이브러리 스택 요약

| 계층 | 선택 | 선정 이유 | 대안 |
|---|---|---|---|
| 오케스트레이션 | **LangChain** (`langchain`, `langchain-core`) | 요구사항에 명시된 `MarkdownHeaderTextSplitter`, Docstore, Retriever 추상화를 모두 제공 | LlamaIndex (`HierarchicalNodeParser`, `AutoMergingRetriever`) |
| 복잡 PDF/DOCX 파싱 | **Docling** | 표 구조 복원 품질, DOCX 동시 지원, CPU에서도 동작 | Marker (GPU, 수식 강점) |
| 고속 PDF 파싱 | **PyMuPDF + pymupdf4llm** | 대용량에서 압도적 속도, Markdown 출력 | pdfplumber |
| Excel | **pandas + openpyxl + tabulate** | 병합 셀 탐지는 openpyxl, 표 변환은 pandas | — |
| DOCX Fallback | **python-docx** | 가볍고 안정적 | — |
| 청킹 | **langchain-text-splitters** + 자체 `TableAwareSplitter` | 헤더 분할 표준 구현 + 표 파편화 방지 커스텀 | — |
| 임베딩 | **BAAI/bge-m3** (sentence-transformers) | 한국어 성능, 로컬 실행, 긴 입력 | OpenAI `text-embedding-3-large`, Voyage |
| Vector DB | **Chroma** | 로컬 영속, 메타데이터 필터, 경량 | Qdrant (네이티브 하이브리드), Milvus |
| Docstore | **LocalFileStore** (kv docstore) | 설치 불필요, 영속 | Redis, PostgreSQL |
| Sparse 검색 | **rank-bm25 + kiwipiepy** | 한국어 형태소 기반 키워드 매칭 | Elasticsearch/OpenSearch (nori) |
| 재순위 (선택) | **bge-reranker-v2-m3** | 다국어 Cross-Encoder | Cohere Rerank |
| LLM | **Claude Haiku 4.5** via 공식 `anthropic` SDK | 빠르고 저렴, 200K 컨텍스트, SDK 내장 재시도 | `claude-sonnet-5-5`, `claude-opus-5-5` (config.yaml) |
| 설정/스키마 | **`config.yaml`** + dataclasses (`rag/config.py`) | 전체 설정을 파일 하나로 관리, 오타 키·모델 비호환 조기 검출, Python 3.9 호환 | pydantic-settings |
| 로깅 | **logging** (표준) | 파일별 파싱 결과·실패 사유 추적 | loguru |

**예상 `requirements.txt` (Task 2에서 확정)**

```
langchain>=0.3
langchain-core
langchain-community
langchain-text-splitters
langchain-chroma
langchain-huggingface
anthropic
chromadb
sentence-transformers
docling
pymupdf
pymupdf4llm
pandas
openpyxl
tabulate
python-docx
rank-bm25
kiwipiepy
watchdog        # 선택
```

---

## 4. Task 2 코드 모듈 매핑 (예정)

요구사항의 "1) Parser, 2) Chunker, 3) Retriever 및 Pipeline" 구분에 맞춘 구조입니다.

```
rag/
├── config.py            # Settings (청크 크기, k, 가중치, 경로, 모델명)
├── schemas.py           # ParsedDocument, RAGAnswer, Citation
├── parsers/             # 1) Parser
│   ├── base.py          #    BaseParser (추상), ParserError
│   ├── pdf_parser.py    #    PDFParser: Docling → PyMuPDF4LLM Fallback
│   ├── excel_parser.py  #    ExcelParser: to_markdown / 서술형 템플릿
│   ├── docx_parser.py   #    DocxParser: Docling → python-docx
│   └── router.py        #    ParserRouter: 확장자 + 사전 검사 기반 라우팅
├── chunking/            # 2) Chunker
│   ├── table_splitter.py#    TableAwareSplitter (헤더 재주입)
│   └── chunker.py       #    ParentChildChunker (헤더 분할, 크기 보정, 태그 주입)
├── indexing/
│   ├── store.py         #    VectorStore(Chroma) + Docstore + BM25 관리, 매니페스트
│   └── korean_tokenizer.py
├── retrieval/           # 3) Retriever
│   └── hybrid.py        #    HybridParentRetriever (Dense+BM25 → RRF → Parent)
├── generation/
│   └── answerer.py      #    인용 프롬프트, 인용 검증
└── pipeline.py          # 3) Pipeline: ingest(folder) / query(question)
```

**예외 처리 지점 (Task 2에서 구현)**

| 지점 | 예외 상황 | 처리 |
|---|---|---|
| Loader | 0바이트, 권한 오류, 잠금 파일 | skip + 사유 기록 |
| PDF Parser | 암호화, 손상, Docling 타임아웃 | Fallback 파서 → 그래도 실패 시 skip |
| Parser 공통 | 파싱 결과가 빈 문자열/공백 | `EmptyDocumentError` → skip |
| Excel Parser | 빈 시트, 헤더 없는 시트 | 시트 단위 skip, 나머지 시트는 계속 처리 |
| Chunker | 헤더가 전혀 없는 문서 | 문서 전체를 하나의 섹션으로 간주 후 크기 보정 |
| Indexing | 임베딩 모델 로드 실패, DB 쓰기 실패 | 해당 파일 트랜잭션 롤백(매니페스트 미갱신) |
| Retriever | 인덱스가 비어 있음, BM25 미구축 | Dense 단독 모드로 강등 + 경고 |
| Generator | API 오류/레이트 리밋 | 재시도(지수 백오프), 최종 실패 시 검색 결과만 반환 |

---

## 5. 확장 경로 (규모가 커질 때)

- **청크 수십만 개 이상:** Chroma + 메모리 BM25 → **Qdrant**(Dense + Sparse 벡터 네이티브 하이브리드) 또는 **OpenSearch**(nori 분석기 + k-NN)로 이전. `HybridParentRetriever` 인터페이스는 유지.
- **다중 사용자/서버:** Docstore를 Redis/PostgreSQL로 교체, 인제스트는 작업 큐(Celery 등)로 분리.
- **품질 측정:** 질문-정답-근거 문서 평가셋을 만들어 `ragas` 등으로 Recall@k, Faithfulness를 측정하고 청크 크기·RRF 가중치를 튜닝.
