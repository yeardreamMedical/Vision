# 새 Pinecone 계정용 수정된 설정
import os
from pinecone import Pinecone, ServerlessSpec

# ===================================================================
# 1. 새 계정 설정 (기존 코드에서 수정할 부분들)
# ===================================================================

# 새 Pinecone 계정 정보
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # 새 계정의 API Key
# PINECONE_ENV는 더 이상 사용하지 않음 (ServerlessSpec에서 직접 지정)

# 새 인덱스명 (기존과 다르게 설정 권장)
INDEX_NAME = "cxr-image-meta-v2"  # 새 이름으로 변경

def initialize_pinecone_v2():
    """새 Pinecone 계정용 초기화 함수"""
    print("\n=== 새 Pinecone 계정 초기화 ===")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경변수가 설정되지 않았습니다!")
    
    # Pinecone 클라이언트 초기화
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 현재 계정의 사용 가능한 regions 확인
    try:
        existing_indexes = pc.list_indexes().names()
        print(f"기존 인덱스: {existing_indexes}")
    except Exception as e:
        print(f"인덱스 목록 조회 오류: {e}")
        return None
    
    # 인덱스 존재 확인 및 생성
    if INDEX_NAME not in existing_indexes:
        print(f"새 인덱스 '{INDEX_NAME}' 생성 중...")
        try:
            # 무료 계정에서 가장 안정적인 설정
            pc.create_index(
                name=INDEX_NAME,
                dimension=512,  # BioViL-T 임베딩 차원
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",           # AWS 클라우드
                    region="us-east-1"     # 가장 안정적인 리전
                )
            )
            print("인덱스 생성 완료!")
            
            # 인덱스 준비 대기 (새 인덱스는 준비 시간이 필요)
            import time
            print("인덱스 준비 중... (30초 대기)")
            time.sleep(30)
            
        except Exception as e:
            print(f"인덱스 생성 오류: {e}")
            # 다른 리전으로 재시도
            try:
                print("us-west-2 리전으로 재시도...")
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=512,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
                print("인덱스 생성 완료! (us-west-2)")
                time.sleep(30)
            except Exception as e2:
                print(f"재시도 실패: {e2}")
                return None
    else:
        print(f"기존 인덱스 '{INDEX_NAME}' 사용")
    
    # 인덱스 연결
    try:
        index = pc.Index(INDEX_NAME)
        
        # 인덱스 상태 확인
        stats = index.describe_index_stats()
        print(f"인덱스 상태: {stats}")
        
        return index
    except Exception as e:
        print(f"인덱스 연결 오류: {e}")
        return None

# ===================================================================
# 2. 배치 크기 조정 및 재시도 로직 추가
# ===================================================================

def safe_upsert_with_retry(index, vectors, max_retries=3, batch_size=50):
    """재시도 로직이 포함된 안전한 업서트 함수"""
    import time
    
    for attempt in range(max_retries):
        try:
            # 배치 크기를 더 작게 조정 (무료 계정 고려)
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch)
                
                # 배치 간 짧은 대기 (Rate limit 방지)
                time.sleep(0.5)
                
            return True
            
        except Exception as e:
            print(f"업서트 시도 {attempt + 1} 실패: {e}")
            
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = (2 ** attempt) * 10  # 지수 백오프
                print(f"{wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                print("최대 재시도 횟수 초과")
                return False
            else:
                time.sleep(5)
    
    return False

# ===================================================================
# 3. 수정된 메인 업로드 함수
# ===================================================================

def process_and_upload_embeddings_v2():
    """새 계정용 수정된 업로드 함수"""
    print("\n=== 새 계정용 이미지 처리 및 업로드 ===")
    
    # 기존 데이터 로드 함수들 그대로 사용
    from embed_image_metadata import load_and_preprocess_data, create_roi_dict, initialize_models, get_image_embedding, create_metadata
    
    # 데이터 로드
    df = load_and_preprocess_data()
    roi_dict = create_roi_dict(df)
    
    # 모델 초기화
    image_model, image_transform = initialize_models()
    
    # 새 Pinecone 초기화
    index = initialize_pinecone_v2()
    if index is None:
        print("Pinecone 초기화 실패")
        return
    
    # 더 작은 배치로 처리 (무료 계정 고려)
    vectors_to_upsert = []
    batch_size = 50  # 배치 크기 줄임
    
    print(f"\n총 {len(roi_dict)}개 이미지 처리 시작...")
    
    success_count = 0
    error_count = 0
    
    from tqdm import tqdm
    from pathlib import Path
    
    for image_id, roi_data in tqdm(roi_dict.items(), desc="이미지 처리 중"):
        # 이미지 경로 생성
        image_path = Path("/Users/solkim/Desktop/projects/medical/project/data/chestxray14/bbox_images") / image_id
        
        # 임베딩 생성
        embedding = get_image_embedding(str(image_path), image_model, image_transform)
        
        if embedding is not None:
            # 메타데이터 생성
            metadata = create_metadata(image_id, roi_data)
            
            # 벡터 데이터 준비
            vector_data = {
                "id": image_id,
                "values": embedding.tolist(),
                "metadata": metadata
            }
            
            vectors_to_upsert.append(vector_data)
            
            # 배치가 찼으면 안전하게 업로드
            if len(vectors_to_upsert) >= batch_size:
                if safe_upsert_with_retry(index, vectors_to_upsert):
                    success_count += len(vectors_to_upsert)
                    print(f"  배치 업로드 완료: {len(vectors_to_upsert)}개 벡터")
                else:
                    error_count += len(vectors_to_upsert)
                    print(f"  배치 업로드 실패: {len(vectors_to_upsert)}개 벡터")
                
                vectors_to_upsert = []
        else:
            error_count += 1
    
    # 남은 벡터들 업로드
    if vectors_to_upsert:
        if safe_upsert_with_retry(index, vectors_to_upsert):
            success_count += len(vectors_to_upsert)
            print(f"  마지막 배치 업로드 완료: {len(vectors_to_upsert)}개 벡터")
        else:
            error_count += len(vectors_to_upsert)
            print(f"  마지막 배치 업로드 실패: {len(vectors_to_upsert)}개 벡터")
    
    print(f"\n=== 처리 완료 ===")
    print(f"성공: {success_count}개")
    print(f"실패: {error_count}개")
    print(f"총 처리: {success_count + error_count}개")
    
    # 최종 인덱스 상태 확인
    try:
        stats = index.describe_index_stats()
        print(f"\n최종 Pinecone 인덱스 상태:")
        print(f"  총 벡터 수: {stats['total_vector_count']}")
        print(f"  인덱스 차원: {stats.get('dimension', 'N/A')}")
    except Exception as e:
        print(f"인덱스 상태 확인 오류: {e}")

# ===================================================================
# 4. 새 계정용 테스트 함수
# ===================================================================

def test_search_v2(query_image_id: str = None, top_k: int = 5):
    """새 계정용 테스트 검색 함수"""
    print(f"\n=== 새 계정 하이브리드 검색 테스트 (top_k={top_k}) ===")
    
    # 새 Pinecone 연결
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    if query_image_id:
        # 기존 테스트 로직과 동일하지만 새 인덱스 사용
        from embed_image_metadata import load_and_preprocess_data, create_roi_dict, initialize_models, get_image_embedding
        
        df = load_and_preprocess_data()
        roi_dict = create_roi_dict(df)
        
        if query_image_id not in roi_dict:
            print(f"오류: {query_image_id}를 찾을 수 없습니다.")
            return

        query_info = roi_dict[query_image_id]
        query_primary_label = query_info[0]['label'] if query_info else 'Unknown'
        
        print(f"\n쿼리 이미지: {query_image_id}")
        print(f"쿼리 레이블: {[roi['label'] for roi in query_info]}")
        print(f"적용할 필터: 'primary_label'이 '{query_primary_label}'인 데이터")
        
        search_filter = {
            "primary_label": {"$eq": query_primary_label}
        }
        
        # 쿼리 임베딩 생성
        image_model, image_transform = initialize_models()
        from pathlib import Path
        query_path = Path("/Users/solkim/Desktop/projects/medical/project/data/chestxray14/bbox_images") / query_image_id
        query_embedding = get_image_embedding(str(query_path), image_model, image_transform)
        
        if query_embedding is None:
            print("쿼리 이미지 처리 실패")
            return
        
        # 검색 수행
        try:
            results = index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=search_filter
            )
            
            print("\n검색 결과:")
            
            if not results['matches']:
                print("  조건에 맞는 결과가 없습니다.")

            for i, match in enumerate(results['matches']):
                metadata = match['metadata']
                print(f"  {i+1}. ID: {match['id']}")
                print(f"     유사도: {match['score']:.4f}")
                print(f"     레이블: {metadata.get('labels', [])}")
                print(f"     주요 레이블: {metadata.get('primary_label', 'N/A')}")
                print()
        except Exception as e:
            print(f"검색 오류: {e}")
    
    else:
        # 전체 통계 출력
        try:
            stats = index.describe_index_stats()
            print("새 인덱스 통계:")
            print(f"  총 벡터 수: {stats['total_vector_count']}")
            print(f"  차원: {stats.get('dimension', 'N/A')}")
        except Exception as e:
            print(f"통계 조회 오류: {e}")

# 실행 함수
if __name__ == "__main__":
    print("=== 새 Pinecone 계정 마이그레이션 ===")
    
    # 먼저 연결 테스트
    try:
        index = initialize_pinecone_v2()
        if index:
            print("Pinecone 연결 성공!")
            
            # 업로드 진행 여부 확인
            user_input = input("\n임베딩 업로드를 시작하시겠습니까? (y/n): ").strip().lower()
            if user_input == 'y':
                process_and_upload_embeddings_v2()
        else:
            print("Pinecone 연결 실패")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()