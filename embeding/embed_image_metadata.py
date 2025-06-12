# X-ray 이미지 임베딩과 메타데이터를 Pinecone에 저장하는 완전한 파이프라인
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from pinecone import Pinecone, ServerlessSpec

# BioViL-T 관련 import
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.image import ImageInferenceEngine

# ===================================================================
# 1단계: 설정 및 초기화
# ===================================================================

# 사용자 설정 부분 - 본인 경로에 맞게 수정하세요
CSV_PATH = '/Users/solkim/Desktop/projects/medical/project/data/chestxray14/BBox_List_2017.csv'
IMAGE_DIR = '/Users/solkim/Desktop/projects/medical/project/data/chestxray14/bbox_images'


# Pinecone 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")  # 기본값 설정
INDEX_NAME = "cxr-image-meta-512"

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 레이블별 한국어 설명 템플릿
LABEL_TEMPLATES = {
    "Atelectasis": "무기폐(atelectasis) 소견이 관찰됩니다.",
    "Cardiomegaly": "심장비대(cardiomegaly) 소견이 관찰되며, 심흉곽비가 증가되어 있습니다.",
    "Effusion": "흉수(pleural effusion)를 시사하는 늑골횡격막각의 둔화가 관찰됩니다.",
    "Infiltrate": "폐 실질에 경계가 불분명한 침윤 혹은 경화(infiltration/consolidation) 음영이 보입니다.",
    "Mass": "종괴(mass)로 의심되는 음영이 관찰되므로, CT 등 정밀 검사가 필요합니다.",
    "Nodule": "폐 결절(pulmonary nodule)이 관찰됩니다.",
    "Pneumonia": "폐렴(pneumonia)에 합당한 폐 경화(consolidation) 소견이 관찰됩니다.",
    "Pneumothorax": "기흉(pneumothorax)을 시사하는 내장쪽 흉막선(visceral pleural line)이 확인됩니다."
}

print("=== X-ray 이미지 임베딩 및 Pinecone 저장 파이프라인 ===")
print(f"디바이스: {DEVICE}")
print(f"이미지 디렉토리: {IMAGE_DIR}")
print(f"CSV 파일: {CSV_PATH}")

# ===================================================================
# 2단계: 데이터 로드 및 전처리
# ===================================================================

def load_and_preprocess_data():
    """CSV 데이터를 로드하고 bbox 정보를 전처리합니다."""
    print("\n2단계: 데이터 로드 및 전처리 중...")
    
    # CSV 로드 (기존 코드 기반)
    column_names = ['Image_Index', 'Finding_Label', 'x', 'y', 'w', 'h', 'ex1', 'ex2', 'ex3']
    df = pd.read_csv(CSV_PATH, header=None, names=column_names)
    df = df.iloc[1:].reset_index(drop=True)  # 헤더 행 제거
    
    # 숫자 컬럼 변환
    numeric_cols = ['x', 'y', 'w', 'h']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # bbox 최대값 계산
    df['x_max'] = df['x'] + df['w']
    df['y_max'] = df['y'] + df['h']
    
    # 불필요한 컬럼 제거
    df = df.drop(['ex1', 'ex2', 'ex3'], axis=1)
    
    print(f"총 {len(df)}개의 bbox 데이터를 로드했습니다.")
    return df

def create_roi_dict(df):
    """이미지별로 ROI 정보를 그룹화합니다."""
    print("ROI 딕셔너리 생성 중...")
    
    def row_to_roi(row):
        return {
            "label": row.Finding_Label,
            "bbox": [int(row.x), int(row.y), int(row.x_max), int(row.y_max)],
            "description": LABEL_TEMPLATES.get(row.Finding_Label, f"{row.Finding_Label} 소견이 관찰됩니다.")
        }
    
    roi_series = (
        df.groupby("Image_Index")
          .apply(lambda d: [row_to_roi(r) for _, r in d.iterrows()], include_groups=False)
    )
    
    roi_dict = roi_series.to_dict()
    print(f"총 {len(roi_dict)}개 이미지의 ROI 정보를 생성했습니다.")
    return roi_dict

# ===================================================================
# 3단계: 이미지 임베딩 생성 함수
# ===================================================================

def get_image_embedding(image_path: str, model, transform) -> np.ndarray:
    """주어진 이미지 경로에 대해 임베딩 벡터를 추출합니다."""
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        print(f"경고: 이미지 파일이 존재하지 않습니다: {image_path}")
        return None
    
    try:
        # 이미지를 1-채널 흑백('L')으로 강제 변환
        pil_image = Image.open(image_path_obj).convert('L')
        image_tensor = transform(pil_image)
        batch_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            model_output = model(batch_tensor)
        
        # 모델 출력에서 임베딩 추출
        embedding_tensor = model_output.img_embedding
        return embedding_tensor.cpu().detach().numpy().squeeze()  # (512,) 형태로 반환
        
    except Exception as e:
        print(f"오류: 이미지 처리 실패 {image_path}: {e}")
        return None

def initialize_models():
    """BioViL-T 모델과 변환기를 초기화합니다."""
    print("\n3단계: BioViL-T 모델 로딩 중...")
    
    # 모델과 변환기 로드
    image_model = get_biovil_t_image_encoder()
    image_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=480)
    
    # 모델을 디바이스로 이동
    image_model.to(DEVICE)
    image_model.eval()  # 평가 모드로 설정
    
    print("모델 로딩 완료!")
    return image_model, image_transform

# ===================================================================
# 4단계: Pinecone 초기화
# ===================================================================

def initialize_pinecone():
    """Pinecone을 초기화하고 인덱스를 생성/연결합니다."""
    print("\n4단계: Pinecone 초기화 중...")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경변수가 설정되지 않았습니다!")
    
    # Pinecone 클라이언트 초기화
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 인덱스 존재 확인 및 생성
    existing_indexes = pc.list_indexes().names()
    
    if INDEX_NAME not in existing_indexes:
        print(f"새 인덱스 '{INDEX_NAME}' 생성 중...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=512,  # BioViL-T 임베딩 차원
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )
        print("인덱스 생성 완료!")
    else:
        print(f"기존 인덱스 '{INDEX_NAME}' 사용")
    
    # 인덱스 연결
    index = pc.Index(INDEX_NAME)
    return index

# ===================================================================
# 5단계: 메타데이터 생성 및 벡터 업서트
# ===================================================================

def create_metadata(image_id: str, roi_data: list) -> dict:
    """이미지에 대한 메타데이터를 생성합니다."""
    # 모든 레이블과 설명 수집
    labels = [roi["label"] for roi in roi_data]
    descriptions = [roi["description"] for roi in roi_data]
    bboxes = [roi["bbox"] for roi in roi_data]
    
    metadata = {
        "image_id": image_id,
        "labels": labels,
        "label_count": len(labels),
        "primary_label": labels[0] if labels else "Unknown",
        "all_descriptions": " ".join(descriptions),
        "bboxes": [",".join(map(str, bb)) for bb in bboxes],  # list[str] 형식으로 변환 (Pinecone 허용)
        "image_path": str(Path(IMAGE_DIR) / image_id)
    }
    
    return metadata

def process_and_upload_embeddings():
    """모든 이미지를 처리하고 Pinecone에 업로드합니다."""
    print("\n5단계: 이미지 처리 및 Pinecone 업로드 시작...")
    
    # 데이터 로드
    df = load_and_preprocess_data()
    roi_dict = create_roi_dict(df)
    
    # 모델 초기화
    image_model, image_transform = initialize_models()
    
    # Pinecone 초기화
    index = initialize_pinecone()
    
    # 배치 처리를 위한 리스트
    vectors_to_upsert = []
    batch_size = 100  # Pinecone 배치 크기
    
    print(f"\n총 {len(roi_dict)}개 이미지 처리 시작...")
    
    success_count = 0
    error_count = 0
    
    for image_id, roi_data in tqdm(roi_dict.items(), desc="이미지 처리 중"):
        # 이미지 경로 생성
        image_path = Path(IMAGE_DIR) / image_id
        
        # 임베딩 생성
        embedding = get_image_embedding(str(image_path), image_model, image_transform)
        
        if embedding is not None:
            # 메타데이터 생성
            metadata = create_metadata(image_id, roi_data)
            
            # 벡터 데이터 준비
            vector_data = {
                "id": image_id,  # 이미지 ID를 벡터 ID로 사용
                "values": embedding.tolist(),  # numpy array를 list로 변환
                "metadata": metadata
            }
            
            vectors_to_upsert.append(vector_data)
            
            # 배치가 찼으면 업로드
            if len(vectors_to_upsert) >= batch_size:
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    success_count += len(vectors_to_upsert)   # 업서트 성공 후에만 카운트
                    print(f"  배치 업로드 완료: {len(vectors_to_upsert)}개 벡터")
                    vectors_to_upsert = []  # 리스트 초기화
                except Exception as e:
                    error_count += len(vectors_to_upsert)
                    print(f"  배치 업로드 오류: {e}")
                    vectors_to_upsert = []
        else:
            error_count += 1
    
    # 남은 벡터들 업로드
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            success_count += len(vectors_to_upsert)
            print(f"  마지막 배치 업로드 완료: {len(vectors_to_upsert)}개 벡터")
        except Exception as e:
            error_count += len(vectors_to_upsert)
            print(f"  마지막 배치 업로드 오류: {e}")
    
    print(f"\n=== 처리 완료 ===")
    print(f"성공: {success_count}개")
    print(f"실패: {error_count}개")
    print(f"총 처리: {success_count + error_count}개")
    
    # 인덱스 상태 확인
    try:
        stats = index.describe_index_stats()
        print(f"\nPinecone 인덱스 상태:")
        print(f"  총 벡터 수: {stats['total_vector_count']}")
        print(f"  인덱스 차원: {stats.get('dimension', 'N/A')}")
    except Exception as e:
        print(f"인덱스 상태 확인 오류: {e}")

# ===================================================================
# 6단계: 검색 테스트 함수 (하이브리드 검색 적용)
# ===================================================================

def test_search(query_image_id: str = None, top_k: int = 5):
    """업로드된 데이터로 하이브리드 검색 테스트를 수행합니다."""
    print(f"\n6단계: 하이브리드 검색 테스트 (top_k={top_k})")
    
    # Pinecone 연결
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    if query_image_id:
        # 특정 이미지로 검색
        # 데이터 로드는 한번만 수행하도록 최적화 가능하지만, 여기서는 함수의 독립성을 위해 유지
        df = load_and_preprocess_data()
        roi_dict = create_roi_dict(df)
        
        if query_image_id not in roi_dict:
            print(f"오류: {query_image_id}를 찾을 수 없습니다.")
            return

        # ==========================================================
        # 하이브리드 검색을 위한 변경점
        # ==========================================================
        
        # 1. 쿼리 이미지의 '주요 레이블'을 가져옵니다.
        #    create_metadata에서 정의한 대로, 첫 번째 레이블을 주요 레이블로 사용합니다.
        query_info = roi_dict[query_image_id]
        query_primary_label = query_info[0]['label'] if query_info else 'Unknown'
        
        print(f"\n쿼리 이미지: {query_image_id}")
        print(f"쿼리 레이블: {[roi['label'] for roi in query_info]}")
        print(f"적용할 필터: 'primary_label'이 '{query_primary_label}'인 데이터") # <--- 필터 정보 출력
        
        # 2. Pinecone 쿼리에 사용할 필터를 생성합니다.
        #    'primary_label' 메타데이터 필드의 값이 쿼리 이미지의 주요 레이블과 일치('$eq')하는 벡터만 검색 대상
        search_filter = {
            "primary_label": {"$eq": query_primary_label}
        } # <--- 필터 딕셔너리 생성
        
        # ==========================================================

        # 쿼리 이미지의 임베딩 생성
        image_model, image_transform = initialize_models()
        query_path = Path(IMAGE_DIR) / query_image_id
        query_embedding = get_image_embedding(str(query_path), image_model, image_transform)
        
        if query_embedding is None:
            print("쿼리 이미지 처리 실패")
            return
        
        # 3. 검색 수행 시 'filter' 파라미터를 추가합니다.
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=search_filter  # <--- 생성한 필터를 쿼리에 적용
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
    
    else:
        # 전체 통계 출력
        stats = index.describe_index_stats()
        print("인덱스 통계:")
        print(f"  총 벡터 수: {stats['total_vector_count']}")
        print(f"  차원: {stats.get('dimension', 'N/A')}")

# ===================================================================
# 메인 실행 함수
# ===================================================================

def main():
    """메인 파이프라인을 실행합니다."""
    print("X-ray 이미지 임베딩 및 Pinecone 저장 파이프라인을 시작합니다...\n")
    
    try:
        # 임베딩 생성 및 업로드
        process_and_upload_embeddings()
        
        # 선택사항: 테스트 검색 수행
        print("\n테스트 검색을 수행하시겠습니까? (y/n)")
        user_input = input().strip().lower()
        
        if user_input == 'y':
            # 첫 번째 이미지로 테스트
            df = load_and_preprocess_data()
            test_image = df['Image_Index'].iloc[0]
            test_search(query_image_id=test_image, top_k=3)
        
        print("\n파이프라인 완료!")
        
    except Exception as e:
        print(f"파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

# 실행
if __name__ == "__main__":
    main()




# ===================================================================
# 메인 실행 부분 (테스트 전용)
# ===================================================================

# 이 코드는 전체 업로드 파이프라인을 건너뛰고 test_search 함수만 바로 실행합니다.
# # 단, 이전에 최소 한번은 전체 파이프라인을 실행해서 Pinecone에 데이터가 업로드되어 있어야 합니다.
# if __name__ == "__main__":
    
#     print("=====================================================")
#     print(">>> 테스트 검색 모드로 실행합니다. (업로드 과정 생략) <<<")
#     print("=====================================================")

#     # 테스트하고 싶은 이미지 파일명을 여기에 직접 입력하세요.
#     # 이전 대화에서 사용한 이미지로 기본 설정되어 있습니다.
#     TEST_IMAGE_ID = "00013118_008.png"

#     # test_search 함수를 직접 호출합니다.
#     # 이전에 수정한 하이브리드 검색 코드가 실행됩니다.
#     try:
#         # 하이브리드 검색 테스트 실행 (레이블: 'Atelectasis')
#         test_search(query_image_id=TEST_IMAGE_ID, top_k=5)

#         # 다른 이미지로도 테스트 해보세요.
#         # print("\n--- 다른 이미지 테스트 ---")
#         # test_search(query_image_id="00002176_005.png", top_k=5) # 레이블: 'Pneumothorax'

#     except Exception as e:
#         print(f"테스트 실행 중 오류 발생: {e}")
#         import traceback
#         traceback.print_exc()
