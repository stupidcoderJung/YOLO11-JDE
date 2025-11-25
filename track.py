from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

"""
# ----------------------------------------------------------------------------------
# [Option A] 수동 제어 모드 (Custom Loop)
# ----------------------------------------------------------------------------------
# Context:
# - 라이브러리가 기본으로 그려주는 박스 말고, '이동 경로(꼬리)'를 예쁘게 그리고 싶을 때 씁니다.
# - OpenCV를 직접 다루기 때문에 코드가 길지만, 시각화를 내 맘대로 커스텀할 수 있습니다.
# ----------------------------------------------------------------------------------

# 1. 모델 로딩
# - task="jde": 일반 탐지가 아니라 '추적+재식별' 모드로 모델을 깨웁니다.
model = YOLO("./../models/yolo11s-jde-tbhs.pt", task="jde")

# 2. 비디오 파일 열기
video_path = "./../videos/MOT17-13.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 진행률 바(tqdm)를 위해 총 프레임 수 계산

# 3. 이동 경로 저장소 (Track History)
# - Why default_dict? 
#   : 처음 보는 ID가 들어와도 에러 없이 자동으로 빈 리스트 []를 만들어줍니다.
# - 비유: "헨젤과 그레텔이 흘린 빵부스러기 저장소"
track_history = defaultdict(lambda: [])

# 4. 프레임 루프 (한 장씩 처리)
with tqdm(total=total_frames, desc="Processing Frames", unit=" frames") as pbar:
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # [핵심!] 추적 실행
            # - persist=True: 
            #   : 아주 중요합니다. 이 옵션이 켜져 있어야 모델이 "아까 그 사람이 이 사람이다"라고 기억합니다.
            #   : 꺼져있으면(False) 매 프레임마다 기억상실증(금붕어) 걸린 것처럼 새로 탐지합니다.
            results = model.track(
                frame,
                tracker="smiletrack.yaml", # 사용할 추적 알고리즘 설정 파일
                persist=True,
                verbose=False # 로그 너무 많이 뜨면 정신없으니 끕니다.
            )

            # 결과 추출 (박스 좌표, ID)
            boxes = results[0].boxes.xywh.cpu() # 중심좌표(x,y), 너비(w), 높이(h)
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 화면에 기본 박스 그리기
            annotated_frame = results[0].plot()

            # [시각화 커스텀] 이동 경로(꼬리) 그리기
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                
                # 현재 위치 추가 (빵부스러기 떨구기)
                track.append((float(x), float(y))) 
                
                # 큐(Queue) 관리: 최근 30 프레임만 기억
                # - 안 지우면 꼬리가 화면 끝까지 남아서 지저분해집니다.
                if len(track) > 30: 
                    track.pop(0)

                # 선 그리기
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 화면 출력
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            pbar.update(1)

            # 'q' 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
"""

# ----------------------------------------------------------------------------------
# [Option B] 자동 실행 모드 (Built-in Mode) -> 현재 실행되는 코드
# ----------------------------------------------------------------------------------
# Context:
# - 복잡한 for문 필요 없이, "이 폴더에 있는 이미지들로 추적 결과 좀 보여줘"라고 시킬 때 씁니다.
# - MOT 챌린지 데이터셋처럼 '이미지가 폴더에 순서대로 들어있는 경우'에 유용합니다.
# ----------------------------------------------------------------------------------

# 1. 모델 로딩
# - 학습된 가중치 파일(.pt) 경로를 정확히 지정해야 합니다.
model = YOLO("./../models/yolo11s-jde-tbhs.pt", task="jde")

# 2. 추적 시작
results = model.track(
    # Source: 비디오 파일이 아니라 '이미지들이 들어있는 폴더'를 지정했습니다.
    # YOLO는 폴더를 주면 알아서 파일명 순서대로(img001, img002...) 영상을 재생하듯 읽어들입니다.
    source="./tracker/evaluation/TrackEval/data/gt/mot_challenge/MOT17/val_half/MOT17-10-FRCNN/img1/",
    
    tracker="jdetracker.yaml",  # JDE 전용 추적 설정 파일
    show=True,      # True: 실행되는 동안 화면에 팝업창을 띄워서 결과를 보여줍니다.
    
    # [Pitfalls - 주의사항]
    # persist=True: 
    # - 이미지 폴더를 비디오처럼 처리할 때 필수입니다.
    # - 이게 없으면 사진 한 장 한 장을 별개의 사건으로 보고 ID를 매번 새로 부여합니다.
    persist=True,   

    # [이미지 해상도 설정]
    # - MOT17 데이터셋 원본 비율을 유지하거나, 모델이 학습된 크기에 맞추기 위해 설정합니다.
    # - (608, 1088): 32의 배수여야 모델이 좋아합니다. (YOLO의 특성)
    imgsz=(608,1088),
)
