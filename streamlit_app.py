"""
📘 보험 약관 RAG 챗봇 - Streamlit 웹 인터페이스
"""

import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from rag_chatbot import RAGChatbot
import json
from pathlib import Path

# 환경변수 로드
load_dotenv()

# 대화 기록 파일 경로
CHAT_HISTORY_FILE = Path("chat_history.json")

def save_chat_history():
    """대화 기록을 JSON 파일로 저장"""
    try:
        chat_data = {
            "last_updated": datetime.now().isoformat(),
            "total_chats": len(st.session_state.chat_history),
            "chat_history": st.session_state.chat_history
        }
        
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.error(f"대화 기록 저장 오류: {str(e)}")
        return False

def load_chat_history():
    """저장된 대화 기록을 불러오기"""
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            # 최근 50개 대화만 로드 (메모리 절약)
            recent_chats = chat_data.get("chat_history", [])[-50:]
            
            return recent_chats, chat_data.get("last_updated", "알 수 없음")
        else:
            return [], "저장된 기록 없음"
    except Exception as e:
        st.error(f"대화 기록 불러오기 오류: {str(e)}")
        return [], "오류 발생"

def clear_chat_history():
    """대화 기록 삭제"""
    try:
        if CHAT_HISTORY_FILE.exists():
            CHAT_HISTORY_FILE.unlink()
        st.session_state.chat_history = []
        return True
    except Exception as e:
        st.error(f"대화 기록 삭제 오류: {str(e)}")
        return False

def export_chat_history():
    """대화 기록을 텍스트 파일로 내보내기"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"chat_export_{timestamp}.txt"
        
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write("=== 보험 약관 챗봇 대화 기록 ===\n")
            f.write(f"내보내기 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 대화 수: {len(st.session_state.chat_history)}\n\n")
            
            for i, chat in enumerate(st.session_state.chat_history, 1):
                f.write(f"--- 대화 {i} ---\n")
                f.write(f"질문: {chat['query']}\n")
                f.write(f"답변: {chat['answer']}\n")
                
                if chat.get('sources'):
                    f.write("참고 문서:\n")
                    for j, source in enumerate(chat['sources'], 1):
                        f.write(f"  {j}. 페이지 {source['page']}: {source['content'][:100]}...\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        return export_file
    except Exception as e:
        st.error(f"대화 기록 내보내기 오류: {str(e)}")
        return None

def initialize_session_state():
    """세션 상태 초기화"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        # 저장된 대화 기록 불러오기
        saved_chats, last_updated = load_chat_history()
        st.session_state.chat_history = saved_chats
        st.session_state.chat_history_loaded = True
        st.session_state.last_updated = last_updated
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def initialize_chatbot():
    """챗봇 초기화"""
    try:
        with st.spinner("🤖 RAG 챗봇을 초기화하는 중..."):
            chatbot = RAGChatbot()
            count = chatbot.get_collection_info()
            
            if count == 0:
                st.error("❌ 벡터 저장소가 비어있습니다. 먼저 데이터를 구축하세요.")
                return None
            
            st.session_state.chatbot = chatbot
            st.session_state.initialized = True
            st.success(f"✅ 챗봇 초기화 완료! (총 {count}개 문서)")
            
    except Exception as e:
        st.error(f"❌ 챗봇 초기화 오류: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="보험 약관 RAG 챗봇",
        page_icon="📘",
        layout="wide"
    )
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 헤더
    st.title("📘 보험 약관 RAG 챗봇")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API 키 확인
        if not os.getenv('OPENAI_API_KEY'):
            st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
            st.info("💡 .env 파일에 OPENAI_API_KEY를 설정하세요.")
            return
        
        # 초기화 버튼
        if st.button("🚀 챗봇 초기화", type="primary"):
            initialize_chatbot()
        
        # 상태 표시
        if st.session_state.initialized:
            st.success("✅ 챗봇 준비 완료")
        else:
            st.warning("⚠️ 챗봇 초기화 필요")
        
        st.markdown("---")
        
        # 대화 기록 관리
        st.header("💾 대화 기록")
        
        # 저장된 대화 기록 정보 표시
        if hasattr(st.session_state, 'chat_history_loaded') and st.session_state.chat_history_loaded:
            st.info(f"📚 저장된 대화: {len(st.session_state.chat_history)}개")
            if hasattr(st.session_state, 'last_updated'):
                st.caption(f"마지막 업데이트: {st.session_state.last_updated}")
        
        # 대화 기록 관리 버튼들
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 저장", help="현재 대화 기록을 저장합니다"):
                if save_chat_history():
                    st.success("✅ 대화 기록이 저장되었습니다!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ 삭제", help="모든 대화 기록을 삭제합니다"):
                if clear_chat_history():
                    st.success("✅ 대화 기록이 삭제되었습니다!")
                    st.rerun()
        
        # 대화 기록 내보내기
        if st.button("📤 내보내기", help="대화 기록을 텍스트 파일로 내보냅니다"):
            export_file = export_chat_history()
            if export_file:
                st.success(f"✅ 대화 기록이 {export_file}로 내보내졌습니다!")
        
        st.markdown("---")
        
        # 🧠 AI 기억 관리 섹션
        if st.session_state.initialized:
            st.header("🧠 AI 기억 관리")
            
            # 기억 저장소 통계
            try:
                memory_stats = st.session_state.chatbot.memory.get_memory_stats()
                st.info(f"🧠 AI 기억: {memory_stats['total_memories']}개 저장됨")
                
                if memory_stats.get('recent_memories'):
                    with st.expander("최근 기억 보기"):
                        for memory in memory_stats['recent_memories'][:5]:
                            st.write(f"📝 {memory['user_query']}")
                            st.caption(f"시간: {memory['timestamp']}")
                
            except Exception as e:
                st.error(f"기억 통계 조회 오류: {str(e)}")
            
            # 기억 관리 버튼들
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧠 기억 내보내기", help="AI 기억을 JSON 파일로 내보냅니다"):
                    try:
                        export_file = st.session_state.chatbot.memory.export_memories()
                        if export_file:
                            st.success(f"✅ AI 기억이 {export_file}로 내보내졌습니다!")
                    except Exception as e:
                        st.error(f"기억 내보내기 오류: {str(e)}")
            
            with col2:
                if st.button("🗑️ 기억 삭제", help="모든 AI 기억을 삭제합니다"):
                    try:
                        if st.session_state.chatbot.memory.clear_all_memories():
                            st.success("✅ 모든 AI 기억이 삭제되었습니다!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"기억 삭제 오류: {str(e)}")
            
            # 기억 검색 테스트
            st.subheader("🔍 기억 검색 테스트")
            search_query = st.text_input("검색할 내용을 입력하세요:", placeholder="예: 보험금 지급")
            
            if st.button("🔍 검색") and search_query:
                try:
                    similar_memories = st.session_state.chatbot.memory.search_similar_conversations(search_query, n_results=3)
                    
                    if similar_memories:
                        st.success(f"🔍 {len(similar_memories)}개의 관련 기억을 찾았습니다!")
                        
                        for i, memory in enumerate(similar_memories, 1):
                            with st.expander(f"기억 {i} - 유사도: {memory['similarity_score']:.2f}"):
                                st.write(f"**질문:** {memory['user_query']}")
                                st.write(f"**답변:** {memory['ai_response'][:200]}...")
                                st.write(f"**시간:** {memory['timestamp']}")
                    else:
                        st.info("관련 기억을 찾을 수 없습니다.")
                        
                except Exception as e:
                    st.error(f"기억 검색 오류: {str(e)}")
        
        st.markdown("---")
        
        # 사용법
        st.header("📖 사용법")
        st.markdown("""
        1. **초기화**: 챗봇 초기화 버튼 클릭
        2. **질문**: 아래 입력창에 질문 입력
        3. **답변**: AI가 약관을 바탕으로 답변
        4. **출처**: 답변에 페이지 번호 표시
        """)
        
        # 예시 질문
        st.header("💡 예시 질문")
        example_questions = [
            "보험금 지급 사유는 무엇인가요?",
            "보험료는 어떻게 납입하나요?",
            "면책 사항이 있나요?",
            "보험 기간은 얼마나 되나요?",
            "해지 시 환급금은 어떻게 되나요?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.user_input = question
    
    # 메인 채팅 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 채팅")
        
        # 채팅 히스토리 표시
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat["query"])
            
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                
                # 출처 정보
                if chat["sources"]:
                    with st.expander(f"📚 참고 문서 ({len(chat['sources'])}개)"):
                        for j, source in enumerate(chat["sources"], 1):
                            st.write(f"**문서 {j}** (페이지 {source['page']})")
                            st.write(source["content"])
                            st.write("---")
        
        # 사용자 입력
        if st.session_state.initialized:
            user_input = st.chat_input("보험 약관에 대해 질문하세요...")
            
            if user_input:
                # 사용자 메시지 표시
                with st.chat_message("user"):
                    st.write(user_input)
                
                # AI 답변 생성 (스트리밍)
                with st.chat_message("assistant"):
                    # 스트리밍 응답을 위한 컨테이너
                    message_placeholder = st.empty()
                    sources_placeholder = st.empty()
                    
                    # 초기 로딩 메시지
                    message_placeholder.markdown("🤔 답변을 생성하는 중...")
                    
                    # 스트리밍 응답 생성
                    full_answer = ""
                    final_sources = []
                    streaming_started = False
                    
                    for chunk in st.session_state.chatbot.chat_streaming(user_input):
                        if not chunk["done"]:
                            # 스트리밍 시작
                            if not streaming_started:
                                streaming_started = True
                                message_placeholder.empty()  # 로딩 메시지 제거
                            
                            # 실시간으로 답변 업데이트 (타이핑 효과)
                            full_answer = chunk["answer"]
                            message_placeholder.markdown(full_answer + "▌")
                            time.sleep(0.01)  # 부드러운 스트리밍을 위한 짧은 지연
                        else:
                            # 최종 완성된 응답
                            full_answer = chunk["answer"]
                            final_sources = chunk["sources"]
                            message_placeholder.markdown(full_answer)
                    
                    # 출처 정보 표시
                    if final_sources:
                        with sources_placeholder.expander(f"📚 참고 문서 ({len(final_sources)}개)"):
                            for i, source in enumerate(final_sources, 1):
                                st.write(f"**문서 {i}** (페이지 {source['page']})")
                                st.write(source["content"])
                                st.write("---")
                
                # 채팅 히스토리에 추가
                st.session_state.chat_history.append({
                    "query": user_input,
                    "answer": full_answer,
                    "sources": final_sources,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 자동으로 대화 기록 저장
                save_chat_history()
        else:
            st.info("👆 먼저 사이드바에서 챗봇을 초기화하세요.")
    
    with col2:
        st.header("📊 시스템 정보")
        
        if st.session_state.initialized:
            # 저장소 정보
            st.subheader("📚 벡터 저장소")
            count = st.session_state.chatbot.get_collection_info()
            st.metric("총 문서 수", count)
            
            # 처리된 데이터 정보
            if os.path.exists("processed_data/약관_processed.json"):
                with open("processed_data/약관_processed.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                st.subheader("📄 처리된 PDF")
                st.metric("총 페이지", len(data))
                
                total_tables = sum(page.get("tables_count", 0) for page in data)
                st.metric("총 표", total_tables)
            
            # 채팅 통계
            st.subheader("💬 채팅 통계")
            st.metric("총 대화 수", len(st.session_state.chat_history))
            
        else:
            st.info("챗봇을 초기화하면 시스템 정보가 표시됩니다.")
        
        # 초기화 버튼
        if st.button("🗑️ 채팅 기록 삭제"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()

