import streamlit as st
import os
import time
from dotenv import load_dotenv
from rag_chatbot import RAGChatbot

# 환경변수 로드
load_dotenv()

# ------------------- 세션 초기화 -------------------
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# ------------------- 챗봇 초기화 -------------------
def initialize_chatbot():
    try:
        chatbot = RAGChatbot()
        count = chatbot.get_collection_info()

        if count == 0:
            st.session_state.initialized = False
            st.error("❌ 벡터 저장소가 비어있습니다. 먼저 데이터를 구축하세요.")
            return None

        st.session_state.chatbot = chatbot
        st.session_state.initialized = True
        st.success(f"✅ 챗봇 초기화 완료! (총 {count}개 문서)")

    except Exception as e:
        st.session_state.initialized = False
        st.error(f"❌ 챗봇 초기화 오류: {str(e)}")
        return None

# ------------------- 페이지 설정 -------------------
st.set_page_config(
    page_title="보험 약관 RAG 챗봇",
    page_icon="📘",
    layout="wide"
)

# ------------------- CSS -------------------
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF !important;
}
.main .block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
}
.chat-container {
    height: calc(100vh - 700px);
    overflow-y: auto;
    padding: 0 1rem;
    display: flex;
    flex-direction: column;
}
.user-bubble, .assistant-bubble {
    max-width: 85%;
    padding: 12px 18px;
    margin: 8px 0;
    border-radius: 16px;
    animation: fadeInUp 0.3s ease-out;
}
.user-bubble {
    align-self: flex-end;
    background: linear-gradient(135deg, #FFA94D, #FF7A00);
    color: white;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.1),
                -2px -2px 8px rgba(255,255,255,0.7);
}
.assistant-bubble {
    align-self: flex-start;
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(230, 180, 120, 0.5);
    box-shadow: 4px 4px 12px rgba(0,0,0,0.08),
                -3px -3px 10px rgba(255,255,255,0.9);
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(8px); }
    100% { opacity: 1; transform: translateY(0); }
}
.chat-container::-webkit-scrollbar {
    width: 5px;
}
.chat-container::-webkit-scrollbar-thumb {
    background: #FF7A00;
    border-radius: 4px;
}
.chat-container::-webkit-scrollbar-thumb:hover {
    background: #FF9400;
}
</style>
""", unsafe_allow_html=True)

# ------------------- 타이틀 -------------------
st.title("📘 현대해상 보험 약관 챗봇")
st.markdown("---")

# ------------------- API Key 체크 -------------------
if not os.getenv('OPENAI_API_KEY'):
    st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
elif not st.session_state.initialized:
    initialize_chatbot()

# ------------------- 사이드바 -------------------
with st.sidebar:
    st.header("⚙️ 설정")
    if st.button("🔄 챗봇 재초기화"):
        st.session_state.initialized = False
        st.session_state.chatbot = None
        initialize_chatbot()
        st.rerun()
    st.markdown("---")
    st.header("📖 사용법")
    st.markdown("질문하면 약관 기반으로 답변을 제공합니다.")
    # ------------------- 시스템 정보 -------------------
    if st.session_state.initialized:
        st.metric("문서 수", st.session_state.chatbot.get_collection_info())
        st.metric("대화 수", len(st.session_state.chat_history))

    if st.button("🗑️ 채팅 초기화"):
        st.session_state.chat_history = []
        st.rerun()
# ------------------- 메인 채팅 영역 -------------------
chat_area = st.container()

with chat_area:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for chat in st.session_state.chat_history:
        st.markdown(f"<div class='user-bubble'>{chat['query']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='assistant-bubble'>{chat['answer']}</div>", unsafe_allow_html=True)

    # 스크롤 자동 내려가기
    st.markdown("""
        <script>
        const chatContainer = window.parent.document.querySelector('.chat-container');
        if(chatContainer){
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------- 입력창 + 스트리밍 -------------------
if st.session_state.initialized:
    user_input = st.chat_input("질문을 입력하세요...")

    if user_input:
        # 1) chat_history에 먼저 추가
        st.session_state.chat_history.append({
            "query": user_input,
            "answer": "",
            "sources": []
        })

        # 2) 화면에 질문 말풍선 바로 렌더
        with chat_area:
            st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

        # 3) 답변 placeholder 생성
        message_placeholder = st.empty()
        full_answer = ""
        final_sources = []

        # 4) 스트리밍
        for chunk in st.session_state.chatbot.chat_streaming(user_input):
            if not chunk["done"]:
                full_answer = chunk["answer"]
                message_placeholder.markdown(f"<div class='assistant-bubble'>{full_answer} ▌</div>", unsafe_allow_html=True)
                time.sleep(0.01)
            else:
                full_answer = chunk["answer"]
                final_sources = chunk.get("sources", [])
                message_placeholder.markdown(f"<div class='assistant-bubble'>{full_answer}</div>", unsafe_allow_html=True)

        # 5) 최종 답변, 출처 chat_history에 저장
        st.session_state.chat_history[-1]["answer"] = full_answer
        st.session_state.chat_history[-1]["sources"] = final_sources

        # 7) 화면 갱신
        st.rerun()
