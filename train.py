# import comet_ml
from ultralytics import YOLO
from datetime import datetime
from functools import partial

# ----------------------------------------------------------------------------------
# [Step 1] 환경 변수 및 스레드 설정 (현재는 주석 처리됨)
# ----------------------------------------------------------------------------------
# Why & Context:
# - 리눅스 서버에서 CPU 코어를 얼마나 쓸지 제한하는 설정들입니다.
# - 보통 PyTorch가 알아서 잘 잡지만, 다른 프로세스와 충돌하거나 CPU 점유율이 
#   비정상적으로 튈 때 이 주석을 풀어서 강제로 '8개 코어만 써라'고 명령합니다.
# ----------------------------------------------------------------------------------
"""
import os
# Set number of threads
N_THREADS = '8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS
"""

# ----------------------------------------------------------------------------------
# [Step 2] 실험 기록 도구 (Comet ML) 초기화
# ----------------------------------------------------------------------------------
# Why & Context:
# - 학습은 며칠씩 걸립니다. 터미널만 보고 있을 순 없죠.
# - Comet ML은 학습 그래프(Loss, mAP 등)를 웹사이트 대시보드로 예쁘게 그려주는 도구입니다.
# - 비유: "비행기 블랙박스 + 실시간 계기판" 역할을 합니다.
# ----------------------------------------------------------------------------------
from ultralytics.utils import SETTINGS
# SETTINGS['comet'] = True  # Ultralytics 내부 설정에서 Comet 사용을 켭니다.
# comet_ml.init()           # Comet ML 서버와 통신 시작 (API Key가 필요할 수 있음)

# MOT(다중 객체 추적) 성능 평가를 위한 커스텀 함수 불러오기
from tracker.evaluation.mot_callback import mot_eval


# ----------------------------------------------------------------------------------
# [Step 3] 모델 로딩 (가장 중요한 부분!)
# ----------------------------------------------------------------------------------
# (1) 'yolo11s-jde.yaml': 일반 YOLO가 아니라, Re-ID 헤드가 달린 JDE 전용 설계도입니다.
# (2) task='jde': Ultralytics 기본 기능이 아니라 커스텀 태스크임을 명시합니다.
# (3) .load(...): 깡통 모델이 아니라, 기본 사물 인식은 할 줄 아는 'yolo11s' 가중치에서 시작합니다.
#     - 비유: "신입사원을 뽑는데, 아예 초짜가 아니라 '인턴 경험 있는' 친구를 데려와서 교육시키는 것(Transfer Learning)"
# ----------------------------------------------------------------------------------
model = YOLO('yolo11s-jde.yaml', task='jde').load('./../models/yolo11s.pt')

# 학습 파라미터 설정
epochs = 30   # 총 30바퀴 공부시킵니다.
batch = 32    # 한 번에 문제집 32쪽씩 풉니다.

# ----------------------------------------------------------------------------------
# [Step 4] 콜백(Callback) 등록
# ----------------------------------------------------------------------------------
# Why & Context:
# - 학습 중간중간에 "추적 성능(MOTA, IDF1 등)"을 검사하고 싶습니다.
# - mot_eval 함수를 바로 실행하는 게 아니라, "validation 끝나면 실행해!"라고 예약 걸어두는 겁니다.
# - period=epochs(30): 매 에폭마다 하면 너무 느리니까, 30번째 에폭(마지막)에만 평가하겠다는 뜻입니다.
# ----------------------------------------------------------------------------------
model.add_callback("on_val_end", partial(mot_eval, period=epochs))


# ----------------------------------------------------------------------------------
# [Step 5] 학습 시작 (The Main Loop)
# ----------------------------------------------------------------------------------
model.train(
    # [로깅 관련]
    project='reid_xps', # 프로젝트 폴더명
    # 로그 파일 이름: "모델명_배치크기_에폭_설명_날짜시간" 순으로 적어서 나중에 파일만 봐도 뭔지 알게 함
    name=f'CH-jde-{batch}b-{epochs}e_TBHS_m075_1280px' + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'),

    # [데이터셋]
    data='crowdhuman.yaml',  # CrowdHuman 데이터셋 경로 설정 파일

    # [하드웨어 설정 - 주의!]
    epochs=epochs,
    batch=batch,
    device=[0,1,2,3,4,5,6,7], # GPU 8개를 병렬로 사용합니다. (엄청난 고사양 세팅입니다)
                              # Tip: GPU가 1개라면 device=[0] 또는 device='0'으로 바꿔야 에러가 안 납니다.

    # [이미지 전처리]
    # bbox_erase=0.1, # (주석됨) 박스 일부를 지워서 가려진 상황을 학습시키는 기법
    imgsz=1280,       # 이미지 크기. 
                      # Trade-off: 640보다 1280이 멀리 있는 작은 사람도 잘 보지만, 메모리(VRAM)를 4배 더 먹습니다.
    
    # [프리징(Freezing) - 튜닝 가이드]
    # freeze=23, # (주석됨) 앞단 레이어(Backbone)는 얼리고(학습 안 하고), 뒷단(Head)만 학습할 때 씁니다.
                 # 처음부터 전체를 다 학습하면 시간이 오래 걸리니 전략적으로 사용합니다.

    # [중요! JDE 특화 설정]
    close_mosaic=0,     
    # - 비유: "모자이크는 프랑켄슈타인 만들기"
    # - 일반 YOLO는 이미지 4장을 잘라 붙여서(Mosaic) 학습하는데, 
    #   Re-ID(신원 식별)에서는 사람 몸이 반토막 나면 '누구인지' 알 수 없게 됩니다.
    # - 그래서 JDE 학습 시에는 이 기능을 반드시 끄거나(0), 아주 조심해서 써야 합니다.

    patience=25,        # 25번 동안 성능이 안 오르면 조기 종료(Early Stopping).
    tracker='jdetracker.yaml',  # Re-ID 기능이 켜진 추적기 설정 파일 사용.

    # [저장 및 출력]
    save=True,          # 학습된 모델 저장 (.pt 파일)
    save_json=True,     # 결과 수치 JSON 저장
    plots=True,         # 결과 그래프 그리기
    verbose=True,       # 진행 상황 자세히 출력
    cache=False,        # RAM에 이미지를 미리 올리지 않음 (데이터가 너무 커서 램 부족 방지)
    
    # [정밀도 설정]
    amp=False,          
    # - AMP(Automatic Mixed Precision): 속도를 위해 연산 정밀도를 약간 낮추는 기능(FP16).
    # - False로 둔 이유: Re-ID 벡터값은 아주 미세한 차이로 사람을 구별해야 합니다.
    #   속도를 포기하더라도 정확한 계산(FP32)을 위해 끈 것으로 보입니다.
)
