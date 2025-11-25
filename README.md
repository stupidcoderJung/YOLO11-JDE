# YOLO11-JDE: 자기 지도(Self-Supervised) Re-ID를 활용한 빠르고 정확한 다중 객체 추적

우리는 실시간 객체 탐지와 자기 지도 학습 기반 재식별(Re-ID)을 결합한 빠르고 정확한 다중 객체 추적(MOT) 솔루션인 **YOLO11-JDE**를 소개합니다. YOLO11s에 전용 Re-ID 브랜치를 통합함으로써, 우리 모델은 각 탐지 객체에 대한 외형 특징을 생성하는 JDE(Joint Detection and Embedding, 탐지와 임베딩 동시 수행)를 수행합니다. Re-ID 브랜치는 탐지 학습과 동시에 완전한 자기 지도(fully self-supervised) 방식으로 학습되므로, 비용이 많이 드는 ID 라벨링 데이터셋이 필요하지 않습니다. 구별력 있는 임베딩 학습을 위해 하드 포지티브(hard positive) 및 세미 하드 네거티브(semi-hard negative) 마이닝 전략을 적용한 트리플렛 손실(triplet loss)을 사용합니다. 데이터 연관(Data association)은 움직임, 외형, 위치 정보를 성공적으로 통합하는 맞춤형 추적 구현을 통해 강화되었습니다. YOLO11-JDE는 MOT17 및 MOT20 벤치마크에서 경쟁력 있는 결과를 달성했으며, 기존 JDE 방법들보다 최대 10배 적은 파라미터를 사용하면서도 더 높은 FPS를 기록했습니다. 이는 우리 방식이 실제 애플리케이션에 매우 매력적인 솔루션임을 보여줍니다.

**참고:** 본 논문은 WACV 2025의 '제5회 실제 환경 감시: 응용 및 도전 과제(5th Real-World Surveillance: Applications and Challenges)' 워크숍 발표작으로 선정되었습니다. 전체 논문은 [arXiv](https://arxiv.org/abs/2501.13710v1)에서 확인하실 수 있습니다.

---

## 주요 특징 (Key Features)

- **실시간 성능**: MOT17 및 MOT20 벤치마크에서 높은 추적 정확도를 유지하면서 경쟁력 있는 FPS를 달성합니다.
- **자기 지도 Re-ID 학습**: 모자이크(Mosaic) 데이터 증강과 하드 및 세미 하드 마이닝 전략을 적용한 트리플렛 손실을 통해, 비용이 많이 드는 ID 라벨링 데이터셋의 필요성을 제거했습니다.
- **맞춤형 데이터 연관**: 가림(occlusion) 상황에 대한 강력한 처리를 포함하여, 객체 추적 성능 향상을 위해 움직임, 외형, 위치 정보를 통합합니다.
- **경량 아키텍처**: 다른 JDE 방식에 비해 최대 10배 적은 파라미터를 사용하여 효율적이며 다양한 애플리케이션에 확장 가능합니다.

---

## 데이터셋 정보 (Dataset Information)

YOLO11-JDE 학습에 사용된 데이터셋은 다음과 같습니다:

1. **CrowdHuman**
   - **설명**: 풍부한 주석이 포함된 다양한 군중 장면을 포함합니다.
   - **다운로드**: 공식 웹사이트에서 다운로드하세요: [https://www.crowdhuman.org](https://www.crowdhuman.org).
   - **참고**: 원본의 학습(training) 및 검증(validation) 분할을 그대로 유지합니다.

2. **MOT17**
   - **설명**: 다중 객체 추적을 위한 시퀀스를 제공합니다. 학습에는 바운딩 박스 주석만 사용됩니다 (트랙 ID는 검증용으로만 사용됨).
   - **다운로드**: 공식 웹사이트에서 다운로드하세요: [https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/).
   - **참고**: 이전 연구들(예: *Towards Real-Time Multi-Object Tracking* 및 *Boost-track: boosting the similarity measure and detection confidence for improved multiple object tracking*)을 따라, 각 학습 시퀀스의 후반부를 사용하고 MOT16 벤치마크와 겹치는 ETH 비디오를 제거하여 검증 세트를 구성했습니다.

두 데이터셋 모두 [YOLO 형식](https://docs.ultralytics.com/datasets/detect/)으로 변환해야 합니다. 재식별(Re-ID) 작업의 검증 및 테스트를 위해, 데이터셋 파일에 원본 MOT17 트랙 ID를 포함하는 추가 열(column)을 추가해야 합니다. 소수의 Re-ID 라벨을 이용한 약한 지도 학습(weak supervision)을 원하지 않는 한, 이 열은 학습 중에는 필요하지 않습니다. CrowdHuman에는 트랙 ID가 없으므로, MOT17 검증 및 테스트 세트에만 추가 트랙 ID를 덧붙인 원래 YOLO 구조를 따릅니다.

사용된 설정 파일은 `crowdhuman.yaml`이며, 병합된 두 데이터셋이 포함된 폴더와 매핑되어야 합니다.

---

## 모델 가중치 다운로드 (Download Model Weights)

YOLO11s-JDE의 사전 학습된 모델 가중치는 [여기서 다운로드](https://drive.google.com/drive/folders/16btXRPikwXOsaItn06p4A8cBhTiIJdoZ?usp=share_link)할 수 있습니다.

---

## 결과 (Results)

### 벤치마크
Private detection 프로토콜 하에서의 **MOT17** 및 **MOT20** 결과입니다:

| 지표 (Metric) | MOT17 | MOT20 |
|----------|-------|-------|
| HOTA     | 56.6  | 53.1  |
| MOTA     | 65.8  | 70.9  |
| IDF1     | 70.3  | 66.4  |
| FPS      | 35.9  | 18.9  |

최신 기술(SOTA)들과 비교했을 때, YOLO11-JDE는 훨씬 적은 파라미터로 우수한 FPS와 경쟁력 있는 추적 정확도를 제공합니다.

---

## 사용자 정의 시나리오에서 성능 향상하기

사용자 정의 시나리오에서 성능을 개선하려면 다음 접근 방식을 고려하세요:

1. **하이퍼파라미터 튜닝** – `ultralytics/cfg/trackers/yolojdetracker.yaml`에서 추적기 설정을 조정하세요. 기본값은 MOT17에 최적화되어 있지만 MOT20에도 잘 적용됩니다.
2. **도메인 드리프트(Domain Drift) 감소** – 적응력을 높이기 위해 데이터셋의 참조 이미지를 바운딩 박스 주석만 사용하여 학습 세트에 통합하세요.
3. **약한 지도 학습(Weakly supervised learning)** – 데이터셋에 트랙 정보가 있는 경우, 바운딩 박스와 트랙 ID 주석을 모두 학습 세트에 포함하세요.
4. **추적기 미세 조정(Fine-tuning)** – 데이터셋에 트랙 정보가 있는 경우, `tracker/finetune/evolve.py`를 사용하여 카메라 설정에 더 잘 맞도록 추적기 하이퍼파라미터를 최적화하세요.

---

## 감사의 글 (Acknowledgements)

이 연구는 다음의 지원을 부분적으로 받았습니다:
- 스페인 프로젝트 PID2022-136436NB-I00.
- ICREA Academia 프로그램 산하 ICREA.
- 바르셀로나 대학교(University of Barcelona)의 마일스톤 연구 프로그램.

YOLO11-JDE 코드는 실시간 객체 탐지 모델을 위한 강력한 기반을 제공하는 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 저장소를 기반으로 합니다.

---

## 인용 (Citation)

연구나 애플리케이션에 YOLO11-JDE가 유용했다면, 저희 논문을 인용해 주세요:

```bibtex
@inproceedings{Erregue_2025,
   title={YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID},
   url={http://dx.doi.org/10.1109/WACVW65960.2025.00092},
   DOI={10.1109/wacvw65960.2025.00092},
   booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
   publisher={IEEE},
   author={Erregue, Iñaki and Nasrollahi, Kamal and Escalera, Sergio},
   year={2025},
   month=feb, pages={776–785}}
```
