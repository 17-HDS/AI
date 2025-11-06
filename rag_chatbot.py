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
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import re
import tiktoken

load_dotenv()

class RAGChatbot:
    def __init__(self):
        # BGE-M3 임베딩 모델 초기화 (검색 시 사용)
        print("🤖 BGE-M3 임베딩 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경 가능
            encode_kwargs={'normalize_embeddings': True}  # 코사인 유사도 최적화
        )
        print("✅ BGE-M3 모델 로딩 완료")
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 임베딩 함수 정의 (ChromaDB 최신 버전 호환)
        # ChromaDB 0.4.16+ 버전에서는 input 파라미터를 사용해야 함
        class BGEEmbeddingFunction:
            def __init__(self, embeddings_model):
                self.embeddings_model = embeddings_model
            
            def name(self):
                """ChromaDB가 요구하는 name 메서드"""
                return "bge-m3"
            
            def __call__(self, input):
                """텍스트 리스트를 임베딩 벡터로 변환 (ChromaDB용)"""
                if isinstance(input, str):
                    input = [input]
                return self.embeddings_model.embed_documents(input)
        
        embedding_function = BGEEmbeddingFunction(self.embeddings)
        
        # 컬렉션 로드 (기존 컬렉션에도 embedding_function 필요)
        self.collection = self.client.get_collection(
            name="insurance_terms",
            embedding_function=embedding_function
        )
        
        # LLM 초기화
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        model_name = os.getenv("GPT_MODEL", "gpt-4o-mini")  # 더 경제적인 모델로 변경
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # 토큰 인코더 초기화
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # 토큰 제한 설정
        self.max_context_tokens = 6000  # 컨텍스트 최대 토큰 수
        self.max_total_tokens = 8000  # 프롬프트 전체 최대 토큰 수
        
        # 처리된 데이터 로드 (키워드 검색용)
        self.load_processed_data()
        
        print(f"📚 컬렉션 정보:")
        print(f"   - 이름: {self.collection.name}")
        print(f"   - 모델: {model_name}")
        print(f"   - 총 문서 수: {self.collection.count()}")
        print(f"   - 로드된 페이지: {len(self.processed_data)}개")
        print(f"   - 최대 컨텍스트 토큰: {self.max_context_tokens}")
    
    def load_processed_data(self):
        """처리된 데이터를 로드하여 키워드 검색에 사용"""
        try:
            with open("processed_data/약관_pages.json", 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
            print(f"✅ 처리된 데이터 로드 완료: {len(self.processed_data)}페이지")
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {str(e)}")
            self.processed_data = []
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        try:
            return len(self.encoding.encode(text))
        except:
            # 대략적으로 4자당 1토큰으로 계산
            return len(text) // 4
    
    def truncate_context(self, context: str, max_tokens: int) -> str:
        """컨텍스트를 토큰 수에 맞게 자르기"""
        tokens = self.encoding.encode(context)
        if len(tokens) <= max_tokens:
            return context
        
        # 토큰 수를 줄임
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)
        return truncated_text + "\n... (내용이 잘렸습니다)"
    
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
        
        # 하이브리드 검색으로 관련 문서 찾기 (k=3으로 축소)
        relevant_docs = self.hybrid_search(query, k=3)
        
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
        
        # 컨텍스트 토큰 수 확인 및 제한
        context_tokens = self.count_tokens(context)
        print(f"📊 컨텍스트 토큰 수: {context_tokens}")
        
        if context_tokens > self.max_context_tokens:
            print(f"⚠️ 컨텍스트가 너무 깁니다. {self.max_context_tokens} 토큰으로 제한합니다.")
            context = self.truncate_context(context, self.max_context_tokens)
        
        # 간소화된 프롬프트
        prompt = f"""당신은 보험 약관 전문 상담사입니다. 문서 내용을 바탕으로 정확하게 답변하세요.

문서 내용:
{context}

질문: {query}

답변 규칙:
1. 문서 내용만을 근거로 답변
2. 구체적이고 명확하게 설명
3. 전문 용어는 쉽게 풀어서 설명
4. 페이지 번호를 언급

답변:"""
        
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
        
        # 하이브리드 검색으로 관련 문서 찾기 (k=3으로 축소)
        relevant_docs = self.hybrid_search(query, k=3)
        
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
        
        # 컨텍스트 토큰 수 확인 및 제한
        context_tokens = self.count_tokens(context)
        print(f"📊 컨텍스트 토큰 수: {context_tokens}")
        
        if context_tokens > self.max_context_tokens:
            print(f"⚠️ 컨텍스트가 너무 깁니다. {self.max_context_tokens} 토큰으로 제한합니다.")
            context = self.truncate_context(context, self.max_context_tokens)
        
        # 간소화된 프롬프트
        prompt = f"""당신은 보험 약관 전문 상담사입니다. 문서 내용을 바탕으로 정확하게 답변하세요.

문서 내용:
{context}

질문: {query}

답변 규칙:
1. 문서 내용만을 근거로 답변
2. 구체적이고 명확하게 설명
3. 전문 용어는 쉽게 풀어서 설명
4. 페이지 번호를 언급

답변:"""
        
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
        
        # 하이브리드 검색으로 관련 문서 찾기 (k=3으로 축소)
        relevant_docs = self.hybrid_search(question, k=3)
        
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
        
        # 컨텍스트 토큰 수 확인 및 제한
        context_tokens = self.count_tokens(context)
        print(f"📊 컨텍스트 토큰 수: {context_tokens}")
        
        if context_tokens > self.max_context_tokens:
            print(f"⚠️ 컨텍스트가 너무 깁니다. {self.max_context_tokens} 토큰으로 제한합니다.")
            context = self.truncate_context(context, self.max_context_tokens)
        
        # 간소화된 프롬프트
        prompt = f"""당신은 보험 약관 전문 상담사입니다. 문서 내용을 바탕으로 정확하게 답변하세요.

문서 내용:
{context}

질문: {question}

답변 규칙:
1. 문서 내용만을 근거로 답변
2. 구체적이고 명확하게 설명
3. 전문 용어는 쉽게 풀어서 설명
4. 페이지 번호를 언급

답변:"""
        
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
