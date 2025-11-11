"""
🔍 Vector Store 성능 측정 벤치마크
임베딩 생성, 저장, 검색 성능을 측정합니다.
"""

import time
import sys
import io
import os
import json
import tracemalloc
from typing import List, Dict
from vector_store import VectorStore
from dotenv import load_dotenv

# Windows 콘솔에서 한글 출력을 위한 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

load_dotenv()

class PerformanceBenchmark:
    """성능 측정 클래스"""
    
    def __init__(self):
        self.results = {}
        
    def measure_embedding_speed(self, vector_store: VectorStore, test_texts: List[str]):
        """임베딩 생성 속도 측정"""
        print("\n" + "="*60)
        print("📊 임베딩 생성 속도 측정")
        print("="*60)
        
        start_time = time.time()
        
        # 배치로 임베딩 생성 측정
        total_chars = 0
        for text in test_texts:
            # 임베딩 함수를 직접 사용하여 속도 측정
            _ = vector_store.embedding_function([text])
            total_chars += len(text)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        avg_time_per_text = elapsed_time / len(test_texts)
        chars_per_second = total_chars / elapsed_time if elapsed_time > 0 else 0
        
        print(f"✅ 총 텍스트 수: {len(test_texts)}개")
        print(f"✅ 총 문자 수: {total_chars:,}자")
        print(f"✅ 총 소요 시간: {elapsed_time:.2f}초")
        print(f"✅ 평균 처리 시간: {avg_time_per_text*1000:.2f}ms/텍스트")
        print(f"✅ 처리 속도: {chars_per_second:.0f}자/초")
        print(f"✅ 처리량: {len(test_texts)/elapsed_time:.2f}텍스트/초")
        
        self.results['embedding'] = {
            'total_texts': len(test_texts),
            'total_chars': total_chars,
            'total_time': elapsed_time,
            'avg_time_per_text': avg_time_per_text,
            'chars_per_second': chars_per_second,
            'throughput': len(test_texts)/elapsed_time
        }
        
        return elapsed_time
    
    def measure_storage_speed(self, vector_store: VectorStore, chunks: List[Dict]):
        """벡터 DB 저장 속도 측정"""
        print("\n" + "="*60)
        print("💾 벡터 DB 저장 속도 측정")
        print("="*60)
        
        # 메모리 추적 시작
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        start_time = time.time()
        
        # 실제 저장 수행
        vector_store.store_in_vector_db(chunks)
        
        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed_time = end_time - start_time
        memory_used = (current_memory - start_memory) / (1024 * 1024)  # MB
        peak_memory_mb = peak_memory / (1024 * 1024)  # MB
        
        chunks_per_second = len(chunks) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"✅ 총 청크 수: {len(chunks):,}개")
        print(f"✅ 총 소요 시간: {elapsed_time:.2f}초")
        print(f"✅ 저장 속도: {chunks_per_second:.2f}청크/초")
        print(f"✅ 평균 처리 시간: {elapsed_time/len(chunks)*1000:.2f}ms/청크")
        print(f"✅ 메모리 사용량: {memory_used:.2f}MB")
        print(f"✅ 최대 메모리 사용량: {peak_memory_mb:.2f}MB")
        
        self.results['storage'] = {
            'total_chunks': len(chunks),
            'total_time': elapsed_time,
            'chunks_per_second': chunks_per_second,
            'avg_time_per_chunk': elapsed_time/len(chunks),
            'memory_used_mb': memory_used,
            'peak_memory_mb': peak_memory_mb
        }
        
        return elapsed_time
    
    def measure_search_speed(self, vector_store: VectorStore, test_queries: List[str], top_k: int = 5):
        """검색 속도 측정"""
        print("\n" + "="*60)
        print("🔍 검색 속도 측정")
        print("="*60)
        
        search_times = []
        results_list = []
        
        for query in test_queries:
            start_time = time.time()
            results = vector_store.search_similar(query, top_k=top_k)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            search_times.append(elapsed_time)
            results_list.append(results)
        
        avg_search_time = sum(search_times) / len(search_times)
        min_search_time = min(search_times)
        max_search_time = max(search_times)
        queries_per_second = 1 / avg_search_time if avg_search_time > 0 else 0
        
        print(f"✅ 테스트 쿼리 수: {len(test_queries)}개")
        print(f"✅ 평균 검색 시간: {avg_search_time*1000:.2f}ms")
        print(f"✅ 최소 검색 시간: {min_search_time*1000:.2f}ms")
        print(f"✅ 최대 검색 시간: {max_search_time*1000:.2f}ms")
        print(f"✅ 검색 처리량: {queries_per_second:.2f}쿼리/초")
        print(f"✅ 결과 수: {top_k}개/쿼리")
        
        # 검색 품질 확인
        total_results = sum(len(r) for r in results_list)
        avg_results = total_results / len(results_list)
        print(f"✅ 평균 반환 결과: {avg_results:.1f}개")
        
        self.results['search'] = {
            'total_queries': len(test_queries),
            'avg_search_time': avg_search_time,
            'min_search_time': min_search_time,
            'max_search_time': max_search_time,
            'queries_per_second': queries_per_second,
            'top_k': top_k,
            'avg_results_per_query': avg_results
        }
        
        return search_times
    
    def measure_collection_info(self, vector_store: VectorStore):
        """컬렉션 정보 조회"""
        print("\n" + "="*60)
        print("📊 컬렉션 정보")
        print("="*60)
        
        start_time = time.time()
        count = vector_store.get_collection_info()
        end_time = time.time()
        
        query_time = end_time - start_time
        
        print(f"✅ 컬렉션 조회 시간: {query_time*1000:.2f}ms")
        
        self.results['collection_info'] = {
            'document_count': count,
            'query_time': query_time
        }
    
    def print_summary(self):
        """성능 요약 출력"""
        print("\n" + "="*60)
        print("📈 성능 측정 요약")
        print("="*60)
        
        if 'embedding' in self.results:
            emb = self.results['embedding']
            print(f"\n[임베딩 생성]")
            print(f"   처리량: {emb['throughput']:.2f}텍스트/초")
            print(f"   속도: {emb['chars_per_second']:.0f}자/초")
        
        if 'storage' in self.results:
            st = self.results['storage']
            print(f"\n[벡터 DB 저장]")
            print(f"   처리량: {st['chunks_per_second']:.2f}청크/초")
            print(f"   메모리: {st['peak_memory_mb']:.2f}MB")
        
        if 'search' in self.results:
            sr = self.results['search']
            print(f"\n[검색 성능]")
            print(f"   평균 검색 시간: {sr['avg_search_time']*1000:.2f}ms")
            print(f"   처리량: {sr['queries_per_second']:.2f}쿼리/초")
        
        if 'collection_info' in self.results:
            ci = self.results['collection_info']
            print(f"\n[컬렉션 정보]")
            print(f"   총 문서 수: {ci['document_count']:,}개")
        
        print("\n" + "="*60)
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """결과를 JSON 파일로 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 성능 측정 결과가 '{filename}'에 저장되었습니다.")
        except Exception as e:
            print(f"\n❌ 결과 저장 오류: {e}")


def create_test_data():
    """테스트용 데이터 생성"""
    # 샘플 텍스트들
    test_texts = [
        "보험금 지급 사유는 무엇인가요?",
        "보험료는 어떻게 납입하나요?",
        "면책 사항이 있나요?",
        "보험 기간은 얼마나 되나요?",
        "해지 시 환급금은 어떻게 되나요?",
        "보험 가입 조건은 무엇인가요?",
        "보험료 납입 방법을 알려주세요.",
        "보험금 청구 절차는 어떻게 되나요?",
        "보험 계약 해지 시 유의사항은 무엇인가요?",
        "보험 계약 갱신은 언제 하나요?",
    ]
    
    # 더 긴 텍스트 샘플 (실제 문서와 유사하게)
    long_texts = [
        "본 약관은 관계 법령 및 내부통제기준에 따른 절차를 거쳐 제공됩니다. 보험계약은 보험회사와 계약자 간에 체결되며, 보험료 납입 및 보험금 지급에 관한 사항을 규정합니다.",
        "보험금 지급 사유는 보험계약에서 정한 사고 발생 시 인정됩니다. 지급 절차는 보험금 청구서와 관련 서류를 제출한 후 심사 과정을 거쳐 지급됩니다.",
        "보험료는 월납, 분기납, 반기납, 연납 방식으로 납입할 수 있으며, 계약서에 명시된 납입 기일까지 납입하여야 합니다. 납입 연체 시에는 계약 해지 등 불이익이 있을 수 있습니다.",
    ] * 10  # 총 30개
    
    return test_texts + long_texts


def main():
    """메인 벤치마크 실행"""
    print("🚀 Vector Store 성능 벤치마크 시작")
    print("="*60)
    
    # 벤치마크 인스턴스 생성
    benchmark = PerformanceBenchmark()
    
    # 1. 벡터 저장소 초기화 (성능 측정용 - 실제 데이터 로드 없이)
    print("\n[1단계] 벡터 저장소 초기화 중...")
    vector_store = VectorStore(collection_name="insurance_terms_benchmark")
    
    # 2. 테스트 데이터 준비
    print("\n[2단계] 테스트 데이터 준비 중...")
    test_texts = create_test_data()
    print(f"   - 테스트 텍스트: {len(test_texts)}개")
    
    # 3. 임베딩 생성 속도 측정
    print("\n[3단계] 임베딩 생성 속도 측정 중...")
    benchmark.measure_embedding_speed(vector_store, test_texts[:10])  # 처음 10개만
    
    # 4. 실제 데이터로 저장 속도 측정 (선택적)
    print("\n[4단계] 실제 데이터 저장 속도 측정 중...")
    json_files = [
        "processed_data/all_pdfs_pages.json",
        "processed_data/약관_pages.json"
    ]
    
    json_file = None
    for file_path in json_files:
        if os.path.exists(file_path):
            json_file = file_path
            break
    
    if json_file:
        print(f"   데이터 파일 발견: {json_file}")
        pages_data = vector_store.load_processed_data(json_file)
        
        if pages_data:
            # 일부 페이지만 사용 (전체 측정은 시간이 오래 걸림)
            sample_pages = pages_data[:10]  # 처음 10페이지만
            print(f"   샘플 페이지 수: {len(sample_pages)}개 (전체 {len(pages_data)}개 중)")
            
            chunks = vector_store.process_all_pages(sample_pages)
            if chunks:
                benchmark.measure_storage_speed(vector_store, chunks)
        else:
            print("   ⚠️ 데이터 로드 실패 - 저장 속도 측정 건너뜀")
    else:
        print("   ⚠️ 처리된 데이터 파일 없음 - 저장 속도 측정 건너뜀")
        print("   (실제 측정을 원하시면 먼저 pdf_preprocessor.py를 실행하세요)")
    
    # 5. 검색 속도 측정
    print("\n[5단계] 검색 속도 측정 중...")
    test_queries = [
        "보험금 지급",
        "보험료 납입",
        "계약 해지",
        "면책 사항",
        "보험 기간",
        "환급금",
        "보험 가입 조건",
        "보험금 청구",
    ]
    benchmark.measure_search_speed(vector_store, test_queries, top_k=5)
    
    # 6. 컬렉션 정보 조회
    benchmark.measure_collection_info(vector_store)
    
    # 7. 요약 및 결과 저장
    benchmark.print_summary()
    benchmark.save_results()
    
    print("\n✅ 벤치마크 완료!")


if __name__ == "__main__":
    main()
