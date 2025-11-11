import chromadb
import sys
import io

# Windows 콘솔에서 한글 출력을 위한 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

client = chromadb.PersistentClient("chroma_db")

# 사용 가능한 컬렉션 목록 확인
collections = client.list_collections()
print("=" * 60)
print("[사용 가능한 컬렉션]")
for col in collections:
    print(f"   - {col.name}")
print("=" * 60)

# 실제 사용 중인 컬렉션 확인
collection_name = "insurance_terms"
try:
    collection = client.get_collection(collection_name)
    print(f"\n[컬렉션 '{collection_name}' 정보]")
    print(f"   - 메타데이터: {collection.metadata}")
    print(f"   - 총 문서 수: {collection.count()}")
    
    # 임베딩 함수 정보 확인
    if hasattr(collection, '_embedding_function'):
        print(f"   - 임베딩 함수: {type(collection._embedding_function).__name__}")
    
    # 샘플 데이터로 임베딩 차원 확인
    sample = collection.peek(limit=1)
    
    print("\n[샘플 데이터]")
    print(f"   - 문서 수: {len(sample.get('documents', []))}")
    if sample.get('documents'):
        print(f"   - 첫 번째 문서 미리보기: {sample['documents'][0][:100]}...")
    
    # 임베딩 차원 확인 (ChromaDB는 기본적으로 임베딩을 저장하지 않음)
    try:
        # 컬렉션 메타데이터에서 임베딩 모델 정보 확인
        print("\n[임베딩 모델 정보]")
        embedding_model = collection.metadata.get("embedding_model", "알 수 없음")
        if embedding_model != "알 수 없음":
            print(f"   - 사용 모델: {embedding_model}")
            if "bge-m3" in embedding_model.lower():
                print("   - 임베딩 차원: 1024 (BGE-M3 기본 차원)")
        else:
            print("   - 코드에서 사용 모델: BAAI/bge-m3 (vector_store.py)")
            print("   - 임베딩 차원: 1024 (BGE-M3 기본 차원)")
        print("   - 참고: ChromaDB는 기본적으로 원본 임베딩 벡터를 저장하지 않습니다.")
    except Exception as embed_error:
        print(f"\n   - 임베딩 정보 확인 중 오류: {embed_error}")
    
except Exception as e:
    print(f"\n[오류] {e}")