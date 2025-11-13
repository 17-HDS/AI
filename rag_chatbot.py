"""
📘 RAGChatbot (ChromaDB 기반 단일 버전)
"""

import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()


class RAGChatbot:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "insurance_terms"):
        """RAG 챗봇 초기화"""
        self.client = chromadb.PersistentClient(path=db_path)

        # 컬렉션 불러오기 (없으면 생성)
        self.collection = self.client.get_collection(collection_name)

        # OpenAI 클라이언트
        self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 마지막 검색된 문서 출처 저장
        self.last_sources = []

    # -----------------------------------------------------
    # 🔍 컬렉션 정보
    # -----------------------------------------------------
    def get_collection_info(self):
        """저장된 문서 개수"""
        try:
            count = self.collection.count()
            return count
        except Exception:
            return 0

    # -----------------------------------------------------
    # 🔍 문서 검색
    # -----------------------------------------------------
    def search_similar_docs(self, query: str, top_k: int = 5) -> str:
        """ChromaDB에서 유사한 문서 검색"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        if not results["documents"]:
            self.last_sources = []
            return ""

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        self.last_sources = [
            {
                "content": doc,
                "page": meta.get("page", "?"),
                "source": meta.get("source", "unknown")
            }
            for doc, meta in zip(docs, metadatas)
        ]

        return "\n\n".join(docs)

    # -----------------------------------------------------
    # 💬 스트리밍 답변 생성
    # -----------------------------------------------------
    def chat_streaming(self, query: str):
        """OpenAI 모델을 이용해 스트리밍 방식으로 답변 생성"""
        try:
            context = self.search_similar_docs(query)

            if not context:
                yield {
                    "answer": "관련된 정보를 찾을 수 없습니다. 다른 표현으로 질문해보세요.",
                    "done": True,
                    "sources": []
                }
                return

            prompt = (
                "다음은 보험 약관의 일부 내용입니다. "
                "이를 참고하여 질문에 답하세요.\n\n"
                f"{context}\n\n"
                f"사용자 질문: {query}"
            )

            # GPT 모델 스트리밍 호출
            stream = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 보험 약관 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )

            answer = ""

            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta else ""
                if delta:
                    answer += delta
                    yield {"answer": answer, "done": False}

            # 스트리밍 종료
            yield {"answer": answer, "done": True, "sources": self.last_sources}

        except Exception as e:
            yield {
                "answer": f"❌ 오류 발생: {str(e)}",
                "done": True,
                "sources": []
            }
