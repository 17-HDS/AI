"""
🎯 최종 RAG 챗봇 시스템
향상된 벡터 저장소와 하이브리드 검색 사용
"""

import os
import json
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re

load_dotenv()

class RAGChatbot:
    def __init__(self):
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 컬렉션 로드
        self.collection = self.client.get_collection("insurance_terms")
        
        # LLM 초기화
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        self.llm = ChatOpenAI(
            model_name=os.getenv("GPT_MODEL", "gpt-4-turbo-preview"), 
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # 처리된 데이터 로드 (키워드 검색용)
        self.load_processed_data()
        
        print(f"📚 컬렉션 정보:")
        print(f"   - 이름: {self.collection.name}")
        print(f"   - 총 문서 수: {self.collection.count()}")
        print(f"   - 로드된 페이지: {len(self.processed_data)}개")
    
    def load_processed_data(self):
        """처리된 데이터를 로드하여 키워드 검색에 사용"""
        try:
            with open("processed_data/약관_pages.json", 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
            print(f"✅ 처리된 데이터 로드 완료: {len(self.processed_data)}페이지")
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {str(e)}")
            self.processed_data = []
    
    def get_collection_info(self):
        """컬렉션 정보 조회"""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            print(f"❌ 컬렉션 정보 조회 오류: {str(e)}")
            return 0
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Streamlit용 채팅 메서드"""
        print(f"\n👤 질문: {query}")
        print("-" * 60)
        
        # 하이브리드 검색으로 관련 문서 찾기
        relevant_docs = self.hybrid_search(query, k=5)
        
        if not relevant_docs:
            return {
                "query": query,
                "answer": "❌ 관련 문서를 찾을 수 없습니다.",
                "sources": []
            }
        
        print(f"✅ {len(relevant_docs)}개 관련 청크 발견")
        
        # 페이지별로 그룹화하여 중복 제거
        page_groups = {}
        for doc in relevant_docs:
            page = doc['metadata'].get('page', 'Unknown')
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc)
        
        print(f"📄 관련 페이지: {len(page_groups)}개")
        for page in sorted(page_groups.keys()):
            chunks_count = len(page_groups[page])
            print(f"   - 페이지 {page}: {chunks_count}개 청크")
        
        # 프롬프트 생성 (페이지별로 그룹화)
        context_parts = []
        for page, docs in page_groups.items():
            page_content = "\n".join([doc['content'] for doc in docs])
            context_parts.append(f"[페이지 {page}]\n{page_content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
당신은 보험 약관 전문 상담사입니다. 사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.

문서 내용:
{context}

질문: {query}

답변 가이드라인:
1. **정확성**: 문서에 명시된 내용만을 바탕으로 답변하세요
2. **구체성**: 관련 조항, 규정, 조건을 구체적으로 설명하세요
3. **출처 명시**: 답변 근거가 되는 페이지 번호를 포함하세요
4. **사용자 중심**: 보험 가입자 관점에서 실용적인 정보를 제공하세요
5. **명확성**: 전문 용어는 쉽게 설명하고, 복잡한 내용은 단계별로 설명하세요

질문 유형별 대응:
- **정의/개념 질문**: 명확한 정의와 적용 범위 설명
- **절차/방법 질문**: 구체적인 단계와 필요 서류 안내
- **조건/자격 질문**: 정확한 조건과 예외 사항 설명
- **금액/보상 질문**: 구체적인 금액과 계산 방법 설명

무의미하거나 불명확한 질문의 경우:
- 질문을 더 구체적으로 해달라고 요청
- 어떤 정보가 필요한지 안내
- 예시 질문을 제시

답변:
"""
        
        try:
            print("🤖 AI 답변 생성 중...")
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            print("✅ 답변 생성 완료")
            
            # 참고 페이지 정보 추가 (간소화)
            reference_pages = sorted(page_groups.keys())
            answer += f"\n\n📄 참고 페이지: {', '.join(map(str, reference_pages))}"
            
            # 출처 정보 생성
            sources = []
            for page, docs in page_groups.items():
                for doc in docs:
                    sources.append({
                        "page": page,
                        "content": doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                    })
            
            return {
                "query": query,
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ 답변 생성 오류: {str(e)}")
            return {
                "query": query,
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": []
            }
    
    def chat_streaming(self, query: str):
        """Streamlit용 스트리밍 채팅 메서드"""
        print(f"\n👤 질문: {query}")
        print("-" * 60)
        
        # 하이브리드 검색으로 관련 문서 찾기
        relevant_docs = self.hybrid_search(query, k=5)
        
        if not relevant_docs:
            yield {
                "query": query,
                "answer": "❌ 관련 문서를 찾을 수 없습니다.",
                "sources": [],
                "done": True
            }
            return
        
        print(f"✅ {len(relevant_docs)}개 관련 청크 발견")
        
        # 페이지별로 그룹화하여 중복 제거
        page_groups = {}
        for doc in relevant_docs:
            page = doc['metadata'].get('page', 'Unknown')
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc)
        
        print(f"📄 관련 페이지: {len(page_groups)}개")
        for page in sorted(page_groups.keys()):
            chunks_count = len(page_groups[page])
            print(f"   - 페이지 {page}: {chunks_count}개 청크")
        
        # 프롬프트 생성 (페이지별로 그룹화)
        context_parts = []
        for page, docs in page_groups.items():
            page_content = "\n".join([doc['content'] for doc in docs])
            context_parts.append(f"[페이지 {page}]\n{page_content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
당신은 보험 약관 전문 상담사입니다. 사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.

문서 내용:
{context}

질문: {query}

답변 가이드라인:
1. **정확성**: 문서에 명시된 내용만을 바탕으로 답변하세요
2. **구체성**: 관련 조항, 규정, 조건을 구체적으로 설명하세요
3. **출처 명시**: 답변 근거가 되는 페이지 번호를 포함하세요
4. **사용자 중심**: 보험 가입자 관점에서 실용적인 정보를 제공하세요
5. **명확성**: 전문 용어는 쉽게 설명하고, 복잡한 내용은 단계별로 설명하세요

질문 유형별 대응:
- **정의/개념 질문**: 명확한 정의와 적용 범위 설명
- **절차/방법 질문**: 구체적인 단계와 필요 서류 안내
- **조건/자격 질문**: 정확한 조건과 예외 사항 설명
- **금액/보상 질문**: 구체적인 금액과 계산 방법 설명

무의미하거나 불명확한 질문의 경우:
- 질문을 더 구체적으로 해달라고 요청
- 어떤 정보가 필요한지 안내
- 예시 질문을 제시

답변:
"""
        
        try:
            print("🤖 AI 답변 생성 중...")
            
            # 스트리밍 응답 생성
            full_answer = ""
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = str(chunk)
                
                full_answer += content
                
                # 스트리밍 데이터 전송
                yield {
                    "query": query,
                    "answer": full_answer,
                    "sources": [],  # 나중에 추가
                    "done": False
                }
            
            print("✅ 답변 생성 완료")
            
            # 참고 페이지 정보 추가
            reference_pages = sorted(page_groups.keys())
            full_answer += f"\n\n📄 참고 페이지: {', '.join(map(str, reference_pages))}"
            
            # 출처 정보 생성
            sources = []
            for page, docs in page_groups.items():
                for doc in docs:
                    sources.append({
                        "page": page,
                        "content": doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                    })
            
            # 최종 완성된 응답 전송
            yield {
                "query": query,
                "answer": full_answer,
                "sources": sources,
                "done": True
            }
            
        except Exception as e:
            print(f"❌ 답변 생성 오류: {str(e)}")
            yield {
                "query": query,
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "done": True
            }
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict]:
        """벡터 유사도 검색"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                documents.append({
                    'content': doc,
                    'metadata': metadata,
                    'score': results['distances'][0][i] if 'distances' in results else 0,
                    'method': 'vector'
                })
            
            return documents
        except Exception as e:
            print(f"❌ 벡터 검색 오류: {str(e)}")
            return []
    
    def keyword_search(self, query: str, k: int = 5) -> List[Dict]:
        """키워드 기반 검색"""
        query_lower = query.lower()
        keyword_matches = []
        
        # 검색 키워드 추출
        keywords = re.findall(r'[\w가-힣]+', query_lower)
        
        for page_data in self.processed_data:
            text = page_data['text'].lower()
            score = 0
            
            # 키워드 매칭 점수 계산
            for keyword in keywords:
                if keyword in text:
                    score += text.count(keyword) * 2  # 키워드 매칭에 더 높은 가중치
            
            if score > 0:
                keyword_matches.append({
                    'content': page_data['text'],
                    'metadata': {
                        'page': page_data['page'],
                        'source': page_data['source']
                    },
                    'score': score,
                    'method': 'keyword'
                })
        
        # 점수순으로 정렬하고 상위 k개 반환
        keyword_matches.sort(key=lambda x: x['score'], reverse=True)
        return keyword_matches[:k]
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """하이브리드 검색 (벡터 + 키워드)"""
        print(f"🔍 하이브리드 검색: '{query}'")
        
        # 벡터 검색
        vector_results = self.vector_search(query, k)
        print(f"   📊 벡터 검색: {len(vector_results)}개 결과")
        
        # 키워드 검색
        keyword_results = self.keyword_search(query, k)
        print(f"   🔤 키워드 검색: {len(keyword_results)}개 결과")
        
        # 결과 병합 및 중복 제거
        all_results = []
        seen_content = set()
        
        # 벡터 검색 결과 추가
        for result in vector_results:
            content_hash = hash(result['content'][:100])  # 첫 100자로 중복 판단
            if content_hash not in seen_content:
                all_results.append(result)
                seen_content.add(content_hash)
        
        # 키워드 검색 결과 추가 (중복 제외)
        for result in keyword_results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen_content:
                all_results.append(result)
                seen_content.add(content_hash)
        
        # 점수순으로 정렬
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"   ✅ 최종 결과: {len(all_results)}개")
        return all_results[:k]
    
    def ask_question(self, question: str) -> str:
        """질문에 대한 답변 생성"""
        print(f"\n👤 질문: {question}")
        print("-" * 60)
        
        # 하이브리드 검색으로 관련 문서 찾기
        relevant_docs = self.hybrid_search(question, k=5)
        
        if not relevant_docs:
            return "❌ 관련 문서를 찾을 수 없습니다."
        
        print(f"✅ {len(relevant_docs)}개 관련 청크 발견")
        
        # 페이지별로 그룹화하여 중복 제거
        page_groups = {}
        for doc in relevant_docs:
            page = doc['metadata'].get('page', 'Unknown')
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc)
        
        print(f"📄 관련 페이지: {len(page_groups)}개")
        for page in sorted(page_groups.keys()):
            chunks_count = len(page_groups[page])
            print(f"   - 페이지 {page}: {chunks_count}개 청크")
        
        # 프롬프트 생성 (페이지별로 그룹화)
        context_parts = []
        for page, docs in page_groups.items():
            page_content = "\n".join([doc['content'] for doc in docs])
            context_parts.append(f"[페이지 {page}]\n{page_content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
당신은 보험 약관 전문 상담사입니다. 사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.

문서 내용:
{context}

질문: {question}

답변 가이드라인:
1. **정확성**: 문서에 명시된 내용만을 바탕으로 답변하세요
2. **구체성**: 관련 조항, 규정, 조건을 구체적으로 설명하세요
3. **출처 명시**: 답변 근거가 되는 페이지 번호를 포함하세요
4. **사용자 중심**: 보험 가입자 관점에서 실용적인 정보를 제공하세요
5. **명확성**: 전문 용어는 쉽게 설명하고, 복잡한 내용은 단계별로 설명하세요

질문 유형별 대응:
- **정의/개념 질문**: 명확한 정의와 적용 범위 설명
- **절차/방법 질문**: 구체적인 단계와 필요 서류 안내
- **조건/자격 질문**: 정확한 조건과 예외 사항 설명
- **금액/보상 질문**: 구체적인 금액과 계산 방법 설명

무의미하거나 불명확한 질문의 경우:
- 질문을 더 구체적으로 해달라고 요청
- 어떤 정보가 필요한지 안내
- 예시 질문을 제시

답변:
"""
        
        try:
            print("🤖 AI 답변 생성 중...")
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            print("✅ 답변 생성 완료")
            
            # 참고 페이지 정보 추가 (간소화)
            reference_pages = sorted(page_groups.keys())
            answer += f"\n\n📄 참고 페이지: {', '.join(map(str, reference_pages))}"
            
            return answer
            
        except Exception as e:
            print(f"❌ 답변 생성 오류: {str(e)}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

def main():
    """메인 프로그램"""
    print("🎯 최종 RAG 챗봇 시스템")
    print("=" * 60)
    
    try:
        chatbot = RAGChatbot()
        
        print(f"\n💬 챗봇이 준비되었습니다! (총 {chatbot.collection.count()}개 문서)")
        print("💡 'quit', 'exit', '종료'를 입력하면 프로그램을 종료합니다.")
        print("=" * 60)
        
        while True:
            question = input("\n👤 질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("👋 챗봇을 종료합니다.")
                break
            
            if not question:
                continue
            
            answer = chatbot.ask_question(question)
            print(f"\n🤖 AI: {answer}")
            
    except Exception as e:
        print(f"❌ 시스템 오류: {str(e)}")

if __name__ == "__main__":
    main()
