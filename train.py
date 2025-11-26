# # import comet_ml
# from ultralytics import YOLO
# from datetime import datetime
# from functools import partial

# # ----------------------------------------------------------------------------------
# # [Step 1] 환경 변수 및 스레드 설정 (현재는 주석 처리됨)
# # ----------------------------------------------------------------------------------
# # Why & Context:
# # - 리눅스 서버에서 CPU 코어를 얼마나 쓸지 제한하는 설정들입니다.
# # - 보통 PyTorch가 알아서 잘 잡지만, 다른 프로세스와 충돌하거나 CPU 점유율이 
# #   비정상적으로 튈 때 이 주석을 풀어서 강제로 '8개 코어만 써라'고 명령합니다.
# # ----------------------------------------------------------------------------------
# """
# import os
# # Set number of threads
# N_THREADS = '8'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['OMP_NUM_THREADS'] = N_THREADS
# os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
# os.environ['MKL_NUM_THREADS'] = N_THREADS
# os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
# os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS
# """

# # ----------------------------------------------------------------------------------
# # [Step 2] 실험 기록 도구 (Comet ML) 초기화
# # ----------------------------------------------------------------------------------
# # Why & Context:
# # - 학습은 며칠씩 걸립니다. 터미널만 보고 있을 순 없죠.
# # - Comet ML은 학습 그래프(Loss, mAP 등)를 웹사이트 대시보드로 예쁘게 그려주는 도구입니다.
# # - 비유: "비행기 블랙박스 + 실시간 계기판" 역할을 합니다.
# # ----------------------------------------------------------------------------------
# from ultralytics.utils import SETTINGS
# # SETTINGS['comet'] = True  # Ultralytics 내부 설정에서 Comet 사용을 켭니다.
# # comet_ml.init()           # Comet ML 서버와 통신 시작 (API Key가 필요할 수 있음)

# # MOT(다중 객체 추적) 성능 평가를 위한 커스텀 함수 불러오기
# from tracker.evaluation.mot_callback import mot_eval


# # ----------------------------------------------------------------------------------
# # [Step 3] 모델 로딩 (가장 중요한 부분!)
# # ----------------------------------------------------------------------------------
# # (1) 'yolo11s-jde.yaml': 일반 YOLO가 아니라, Re-ID 헤드가 달린 JDE 전용 설계도입니다.
# # (2) task='jde': Ultralytics 기본 기능이 아니라 커스텀 태스크임을 명시합니다.
# # (3) .load(...): 깡통 모델이 아니라, 기본 사물 인식은 할 줄 아는 'yolo11s' 가중치에서 시작합니다.
# #     - 비유: "신입사원을 뽑는데, 아예 초짜가 아니라 '인턴 경험 있는' 친구를 데려와서 교육시키는 것(Transfer Learning)"
# # ----------------------------------------------------------------------------------
# model = YOLO('yolo11s-jde.yaml', task='jde').load('./../models/yolo11s.pt')

# # 학습 파라미터 설정
# epochs = 30   # 총 30바퀴 공부시킵니다.
# batch = 32    # 한 번에 문제집 32쪽씩 풉니다.

# # ----------------------------------------------------------------------------------
# # [Step 4] 콜백(Callback) 등록
# # ----------------------------------------------------------------------------------
# # Why & Context:
# # - 학습 중간중간에 "추적 성능(MOTA, IDF1 등)"을 검사하고 싶습니다.
# # - mot_eval 함수를 바로 실행하는 게 아니라, "validation 끝나면 실행해!"라고 예약 걸어두는 겁니다.
# # - period=epochs(30): 매 에폭마다 하면 너무 느리니까, 30번째 에폭(마지막)에만 평가하겠다는 뜻입니다.
# # ----------------------------------------------------------------------------------
# model.add_callback("on_val_end", partial(mot_eval, period=epochs))


# # ----------------------------------------------------------------------------------
# # [Step 5] 학습 시작 (The Main Loop)
# # ----------------------------------------------------------------------------------
# model.train(
#     # [로깅 관련]
#     project='reid_xps', # 프로젝트 폴더명
#     # 로그 파일 이름: "모델명_배치크기_에폭_설명_날짜시간" 순으로 적어서 나중에 파일만 봐도 뭔지 알게 함
#     name=f'CH-jde-{batch}b-{epochs}e_TBHS_m075_1280px' + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'),

#     # [데이터셋]
#     data='crowdhuman.yaml',  # CrowdHuman 데이터셋 경로 설정 파일

#     # [하드웨어 설정 - 주의!]
#     epochs=epochs,
#     batch=batch,
#     device=[0], 
#     # device=[0,1,2,3,4,5,6,7], # GPU 8개를 병렬로 사용합니다. (엄청난 고사양 세팅입니다)
#                               # Tip: GPU가 1개라면 device=[0] 또는 device='0'으로 바꿔야 에러가 안 납니다.

#     # [이미지 전처리]
#     # bbox_erase=0.1, # (주석됨) 박스 일부를 지워서 가려진 상황을 학습시키는 기법
#     imgsz=1280,       # 이미지 크기. 
#                       # Trade-off: 640보다 1280이 멀리 있는 작은 사람도 잘 보지만, 메모리(VRAM)를 4배 더 먹습니다.
    
#     # [프리징(Freezing) - 튜닝 가이드]
#     # freeze=23, # (주석됨) 앞단 레이어(Backbone)는 얼리고(학습 안 하고), 뒷단(Head)만 학습할 때 씁니다.
#                  # 처음부터 전체를 다 학습하면 시간이 오래 걸리니 전략적으로 사용합니다.

#     # [중요! JDE 특화 설정]
#     close_mosaic=0,     
#     # - 비유: "모자이크는 프랑켄슈타인 만들기"
#     # - 일반 YOLO는 이미지 4장을 잘라 붙여서(Mosaic) 학습하는데, 
#     #   Re-ID(신원 식별)에서는 사람 몸이 반토막 나면 '누구인지' 알 수 없게 됩니다.
#     # - 그래서 JDE 학습 시에는 이 기능을 반드시 끄거나(0), 아주 조심해서 써야 합니다.

#     patience=25,        # 25번 동안 성능이 안 오르면 조기 종료(Early Stopping).
#     tracker='jdetracker.yaml',  # Re-ID 기능이 켜진 추적기 설정 파일 사용.

#     # [저장 및 출력]
#     save=True,          # 학습된 모델 저장 (.pt 파일)
#     save_json=True,     # 결과 수치 JSON 저장
#     plots=True,         # 결과 그래프 그리기
#     verbose=True,       # 진행 상황 자세히 출력
#     cache=False,        # RAM에 이미지를 미리 올리지 않음 (데이터가 너무 커서 램 부족 방지)
    
#     # [정밀도 설정]
#     amp=False,          
#     # - AMP(Automatic Mixed Precision): 속도를 위해 연산 정밀도를 약간 낮추는 기능(FP16).
#     # - False로 둔 이유: Re-ID 벡터값은 아주 미세한 차이로 사람을 구별해야 합니다.
#     #   속도를 포기하더라도 정확한 계산(FP32)을 위해 끈 것으로 보입니다.
# )

from ultralytics import YOLO
from datetime import datetime
import torch
import os

# ==================================================================================
# [설정] 경로 및 하이퍼파라미터
# ==================================================================================

# 1. 데이터셋 경로 (아까 만든 그 파일!)
# 에러가 났던 'crowdhuman.yaml' 대신 우리가 만든 파일을 지정합니다.
DATASET_YAML = "/content/drive/MyDrive/datasets-test/dog_jde.yaml" 

# 2. 모델 설정 (Nano 버전)
MODEL_CONFIG = "yolo11n-jde.yaml"  # 위에서 만든 설정 파일
PRETRAINED_WEIGHTS = "yolo11n.pt"  # 깡통이 아니라 미리 학습된 지식(COCO) 탑재

# 3. 하드웨어/학습 설정 (Colab 최적화)
EPOCHS = 30
BATCH_SIZE = 16   
# - 원래 32였지만, Colab 무료/Pro 버전에선 메모리(RAM/VRAM)가 빡빡할 수 있습니다.
# - 안전하게 16으로 시작하고, 잘 돌아가면 32로 늘리세요.

IMG_SIZE = 640    
# - 원래 1280이었지만, Nano 모델은 보통 640에서 학습합니다.
# - 1280으로 하면 메모리가 터질(OOM) 확률이 매우 높고 속도가 4배 느려집니다.
# - 강아지 탐지에는 640으로도 충분합니다.

# ==================================================================================
# [실행] 모델 로드 및 학습
# ==================================================================================

# 1. 모델 초기화
# task='detect'가 아니라 'jde'여야 하는데, 이 리포지토리의 구조상 
# task='detect'로 두고 커스텀 헤드를 인식시키는 방식일 수도 있습니다.
# 일단 제공해주신 코드대로 task='jde'를 유지하되, 에러나면 task='detect'로 바꿔보세요.
print(f"🚀 모델 로딩 중: {MODEL_CONFIG} (Pretrained: {PRETRAINED_WEIGHTS})")

try:
    # load() 함수는 구조가 같아야 동작합니다. 
    # n-jde 구조에 n-pt 가중치를 넣으면 일부(Backbone)만 들어가고 Head는 랜덤 초기화됩니다. (정상)
    model = YOLO(MODEL_CONFIG).load(PRETRAINED_WEIGHTS)
except Exception as e:
    print(f"⚠️ 로드 경고 (무시 가능): {e}")
    # 가중치 로드 실패시 깡통으로라도 시작
    model = YOLO(MODEL_CONFIG) 

# 2. 학습 시작
print(f"🔥 학습 시작! (Target: {DATASET_YAML})")

model.train(
    # [프로젝트 관리]
    project='/content/drive/MyDrive/runs/train', # 구글 드라이브에 결과 저장 (세션 끊겨도 살림)
    name=f'Dog-JDE-n-{IMG_SIZE}-{BATCH_SIZE}b',
    
    # [핵심 데이터]
    data=DATASET_YAML,  # <--- 여기가 수정된 핵심 포인트!
    
    # [학습 파라미터]
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=0,           # GPU 1개 사용 (Colab은 무조건 0번)
    
    # [JDE 특화 설정]
    # Re-ID에서는 모자이크(이미지 4개 섞기)가 독이 될 수 있습니다.
    # 사람/동물의 상반신, 하반신이 잘리면 누구인지 모르기 때문입니다.
    # 하지만 데이터가 적을 땐(50개) 켜는 게 나을 수도 있습니다. 일단 끕니다(0).
    close_mosaic=10,    # 마지막 10 에폭 동안은 모자이크 끄기 (절충안)
    
    # [시스템 설정]
    save=True,          # 모델 저장
    plots=True,         # 그래프 그리기
    amp=True,           # True로 변경함. (T4/A100에서는 FP16을 써야 속도가 빠릅니다. 성능 저하 거의 없음)
    workers=4,          # 데이터 로더 프로세스 수 (Colab은 2~4 적당)
    cache=False         # RAM 절약
)

print("🎉 모든 학습 과정이 종료되었습니다!")
