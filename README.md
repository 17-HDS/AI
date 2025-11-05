# 📘 보험 약관 RAG 챗봇 시스템

보험 약관 PDF 파일을 벡터 데이터베이스에 저장하고, GPT API를 이용하여 실제 약관에 기반한 질문-응답이 가능한 AI 챗봇 시스템입니다.

## 🏗️ 시스템 아키텍처

```
📘 보험 약관 RAG 챗봇 시스템
├── Step 1: PDF 전처리 (pdf_preprocessor.py)
│   ├── PyMuPDF: 전체 텍스트 추출
│   ├── pdfplumber: 표 인식 및 추출
│   ├── 출력: JSON(벡터DB용) + TXT(검증용) + XLSX(표 전용)
│   └── 저장: processed_data/ 폴더
│
├── Step 2: 벡터 저장소 (vector_store.py)
│   ├── LangChain: 텍스트 청킹 (300토큰, 100오버랩)
│   ├── OpenAI: 임베딩 (text-embedding-ada-002)
│   └── ChromaDB: 벡터 저장
│
├── Step 3: RAG 챗봇 (rag_chatbot.py)
│   ├── 검색: 하이브리드 검색 (벡터 + 키워드)
│   ├── 생성: GPT-4 Turbo 답변 생성
│   └── 출처: 페이지 번호 명시
│
└── 인터페이스 (streamlit_app.py)
    ├── 웹 UI: Streamlit 기반
    ├── 채팅: 실시간 질의응답
    └── 출처: 참고 문서 표시
```

## 🚀 빠른 시작

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 설정
`.env` 파일 생성:
```env
# OpenAI API 키 (필수)
OPENAI_API_KEY=sk-your-api-key-here

# 사용할 GPT 모델 (선택사항, 기본값: gpt-4o-mini)
# 추천 모델:
# - gpt-4o-mini: 가장 경제적이고 빠름 (추천) ✨
# - gpt-4o: 균형잡힌 성능과 비용
# - gpt-4-turbo: 높은 품질, 높은 비용 (rate limit 주의)
# - gpt-3.5-turbo: 빠르고 저렴함
GPT_MODEL=gpt-4o-mini
```

> ⚠️ **Rate Limit 오류 해결**: 
> `rate_limit_exceeded` 오류가 발생하면 GPT_MODEL을 `gpt-4o-mini`로 변경하세요. 
> 이 모델은 더 높은 토큰 제한과 저렴한 비용을 제공합니다.

### 3. PDF 파일 준비
`source/` 폴더에 `약관.pdf` 파일을 넣으세요.

### 4. 단계별 실행
```bash
# 1단계: PDF 전처리
python pdf_preprocessor.py

# 2단계: 벡터 저장소 구축
python vector_store.py

# 3단계: 웹 인터페이스 실행
streamlit run streamlit_app.py

## 📁 프로젝트 구조

```
프로젝트 폴더/
├── source/                    # PDF 파일 저장
│   └── 약관.pdf
├── processed_data/            # 처리된 데이터 (자동 생성)
│   ├── 약관_pages.json       # 벡터DB용 페이지별 텍스트
│   ├── 약관_extracted.txt    # 검증용 텍스트
│   └── 약관_tables.xlsx      # 표 전용 엑셀
├── chroma_db/                # 벡터 저장소 (자동 생성)
├── pdf_preprocessor.py       # PDF 전처리
├── vector_store.py           # 벡터 저장소 관리
├── rag_chatbot.py           # RAG 챗봇
├── streamlit_app.py         # 웹 인터페이스
├── requirements.txt         # 패키지 목록
├── .env                     # API 키 설정
└── README.md               # 이 파일
```

## 🔧 개별 실행

### PDF 전처리만 실행
```bash
python pdf_preprocessor.py
```

### 벡터 저장소만 구축
```bash
python vector_store.py
```

### 콘솔 챗봇만 실행
```bash
python rag_chatbot.py
```

## 💡 주요 기능

- **완전한 PDF 처리**: 텍스트 + 표 통합 추출 (JSON, TXT, XLSX 3중 저장)
- **스마트 청킹**: 300토큰 단위로 의미 단위 분할 (100토큰 오버랩)
- **하이브리드 검색**: 벡터 유사도 + 키워드 매칭 결합
- **출처 명시**: 모든 답변에 페이지 번호 포함
- **웹 인터페이스**: 사용자 친화적 UI
- **실시간 채팅**: 자연어 질의응답
- **표 전용 저장**: 엑셀 파일로 표 데이터 별도 관리

## 📋 예시 질문

- "보험금 지급 사유는 무엇인가요?"
- "보험료는 어떻게 납입하나요?"
- "면책 사항이 있나요?"
- "보험 기간은 얼마나 되나요?"
- "해지 시 환급금은 어떻게 되나요?"

## ⚙️ 기술 스택

- **PDF 처리**: PyMuPDF, pdfplumber
- **텍스트 처리**: LangChain, tiktoken
- **임베딩**: OpenAI text-embedding-ada-002
- **벡터 DB**: ChromaDB
- **LLM**: GPT-4o-mini (기본값) / GPT-4 Turbo / GPT-3.5-turbo
- **웹 UI**: Streamlit

## 💰 비용 최적화

본 시스템은 다음과 같은 방법으로 비용을 최적화합니다:

1. **토큰 제한**: 
   - 컨텍스트 최대 6,000 토큰으로 제한
   - 검색 결과 수를 3개로 제한
   - 프롬프트 템플릿 간소화

2. **경제적 모델 사용**:
   - 기본 모델: `gpt-4o-mini` (가장 저렴하고 빠름)
   - 필요시 `.env`에서 GPT_MODEL 변경 가능

3. **스마트 청킹**:
   - 300토큰 단위로 분할하여 효율적인 검색

## 🔍 문제 해결

### API 키 오류
- `.env` 파일에 올바른 OpenAI API 키가 설정되어 있는지 확인

### Rate Limit 오류 (429 Error)
**증상**: `Error code: 429 - rate_limit_exceeded` 오류 발생

**원인**: 
- 토큰 사용량이 API 제한을 초과
- `gpt-4-turbo-preview`의 경우 30,000 TPM 제한

**해결 방법**:
1. `.env` 파일에서 모델 변경:
   ```env
   GPT_MODEL=gpt-4o-mini
   ```
2. 더 경제적이고 높은 제한의 모델 사용
3. 시스템이 자동으로 컨텍스트를 6,000 토큰으로 제한

### 벡터 저장소 오류
- `python vector_store.py`를 먼저 실행하여 데이터를 구축
- `processed_data/약관_pages.json` 파일이 있는지 확인

### PDF 파일 오류
- `source/` 폴더에 `약관.pdf` 파일이 있는지 확인

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 모든 패키지가 설치되었는지
2. API 키가 올바르게 설정되었는지
3. PDF 파일이 올바른 위치에 있는지