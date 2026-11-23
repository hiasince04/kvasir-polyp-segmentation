# 내시경 용종 분할의 실패 분석과 검수 우선순위

**메디인테크 AI Research Engineer 동계 현장실습 지원 연구 프로젝트**

Kvasir-SEG에서 경량 용종 분할 모델을 학습하고, 실패가 집중되는 크기와 경계 특성을 분석합니다. 정답 마스크가 없는 환경에서 계산한 검수 점수가 같은 예산의 무작위 검수보다 오류를 더 많이 찾는지 평가합니다. 검증 데이터에서 선택한 한 가지 손실 변경은 고정 테스트셋에서 비교합니다.

## Colab에서 실행

1. `Kvasir_SEG_Research.ipynb`를 [Google Colab](https://colab.research.google.com/)에 업로드합니다.
2. **런타임 → 런타임 유형 변경**에서 GPU를 선택합니다.
3. **런타임 → 모두 실행**을 누릅니다. 기본값인 `PROFILE="research"`는 공식 데이터와 사전학습 가중치를 받습니다.
4. 마지막 셀에서 `RESULTS.md` 및 결과 ZIP의 경로를 확인합니다. 자동 다운로드가 시작되지 않으면 Colab의 **파일** 탭에서 ZIP을 내려받습니다.

기본 연구 설정은 256×256 입력, 배치 8, 두 모델에 각각 12 epoch입니다. 데이터 전송 속도와 할당 GPU에 따라 실행 시간이 달라집니다. `PROFILE="smoke"`는 합성 이미지로 전체 셀 순서와 결과 저장을 확인하는 별도 프로파일입니다.

## 데이터 출처와 이용 조건

Simula 공식 Kvasir-SEG는 용종 영상 1,000장, 파일명이 대응하는 분할 마스크 1,000장, bounding box 정보를 제공합니다. 논문과 공식 배포 페이지의 구조를 확인했으며, 구현은 `images/`와 `masks/`의 파일명을 대조합니다. 주 배포본의 체크섬 확인 후 압축을 풉니다. Simula의 두 경로가 실패하면 원저자 Debesh Jha가 게시한 Kaggle v3 배포본에서 같은 영상·마스크 폴더를 읽습니다. 사용한 URL과 ZIP의 SHA-256을 `run_setup.json`에 기록합니다. 원저자의 Kaggle ZIP은 별도 패키징이라 Simula ZIP과 파일 해시가 다릅니다. [Simula 공식 배포](https://datasets.simula.no/kvasir-seg/), [Simula ZIP](https://datasets.simula.no/downloads/kvasir-seg.zip), [Kaggle 저자 배포본](https://www.kaggle.com/datasets/debeshjha1/kvasirseg), [Simula ZIP의 공개 체크섬 기록](https://constantinpape.github.io/torch-em/torch_em/data/datasets/medical/kvasir.html)

공식 이용 조건은 연구·교육 목적의 사용을 허용하며, 상업적 사용에는 사전 서면 허가가 필요합니다. 실험에 데이터셋을 사용하거나 결과를 보고할 때 아래 논문을 인용합니다. 제공하는 소스 ZIP에는 원본 데이터, 모델 가중치, 실행 산출물이 들어 있지 않습니다. Colab이 생성하는 결과 ZIP에는 평가용 사례 그림과 학습 가중치가 들어 있습니다. [공식 이용 조건](https://datasets.simula.no/kvasir-seg/)

## 실험 설계

| 단계 | 데이터 | 사용 목적 |
| --- | --- | --- |
| 모델 학습 | train, 약 70% | 가중치 업데이트, 마스크 크기·경계 삼분위 기준 계산 |
| 모델 선택 | val_select, 약 10% | 두 모델 각각에서 macro Dice 최고 epoch 선택 |
| 실패 및 검수 분석 | val_audit, 약 10% | 실패 유형 분석, 개선 가설 선택, 검수 실험과 합성 오류 주입 |
| 최종 평가 | test, 약 10% | 확정된 두 체크포인트의 짝지은 비교 |

이미지의 디코딩 RGB SHA-256이 같거나, 128비트 dHash 거리 ≤ 6 및 32×32 RGB 평균 절대 차이 ≤ 0.08을 함께 만족하면 중복 후보로 연결합니다. 연결 성분 전체를 한 분할에 배정합니다. 분할 생성에는 이미지와 파일명만 사용합니다. 공식 공개본에는 환자·검사 세션 ID가 없어 환자 단위 독립성을 검증할 수 없습니다. 해시 기준에서 누락되는 유사 장면도 남을 수 있습니다.

### 기준 모델과 실패 분석

- TorchVision LR-ASPP MobileNetV3-Large의 `COCO_WITH_VOC_LABELS_V1` 분할 가중치에서 시작해 두 분류기를 1채널로 교체합니다. 기본 손실은 BCE와 soft Dice loss의 평균입니다. [TorchVision 모델 API](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html), [v0.24.1 구현](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/segmentation/lraspp.py)
- 종횡비를 유지해 리사이즈하고 RGB는 bilinear, 마스크는 nearest-neighbor로 보간합니다. 마스크의 휘도 ≥ 128은 전경이며, 패딩은 손실과 평가에서 제외합니다. 좌우·상하 반전과 밝기 증강은 학습에만 적용합니다.
- 임계값 0.50의 예측으로 이미지별 Dice, IoU, Recall, Precision을 구한 뒤 macro 평균을 냅니다. 예측과 참조가 모두 비면 Dice·IoU·Precision은 1, 예측만 비고 참조가 있으면 Precision은 0, 참조가 비면 Recall은 1로 정의합니다. 풀 해상도 기준의 점수와 직접 비교할 때는 차이를 고려해야 합니다.
- 검증 실패는 Dice < 0.60입니다. 크기는 용종 픽셀 비율, 경계 복잡도는 전처리 격자에서 계산한 둘레²/(4π×면적)입니다. 경계 접촉과 연결 성분 수도 집계합니다. Boundary F1은 예측·참조 윤곽의 최근접 거리가 2픽셀 이내일 때 매칭합니다. 형태별 차이는 탐색적 연관으로 해석합니다.

### 정답 없는 검수와 오프라인 평가

| 실험 | 순위 생성 입력 | 오프라인 평가 정답 |
| --- | --- | --- |
| 분할 예측 실패 | RGB 영상의 4회 반전 TTA 예측 엔트로피 상위 5% 평균 | 기준 모델 단일 추론의 Dice < 0.60 |
| 주석 오류 모의실험 | 후보 마스크와 RGB 모델 출력의 확신도·엔트로피 가중 불일치 | 원본 마스크와 후보 간 실제 변경 여부 및 주입 유형 |

순위를 먼저 저장한 뒤 평가 레이블을 조인합니다. 10·20·30·50% 검수 예산에서 발견 수, 오류 재현율, 검수 정밀도와 무작위 평균 대비 배수를 계산합니다. 무작위 비교는 같은 이미지, 같은 검수 건수에서 500개 순열을 뽑습니다. 자연 실패의 곡선에 표시한 구간은 무작위 순열 결과의 2.5–97.5 백분위입니다.

합성 실험은 12회 반복하며 매번 분석셋의 절반에 이동·침식·팽창·삭제를 균형 있게 배정합니다. 전경 변경이 발생하지 않은 주입은 음성으로 집계합니다. 가중 점수는 모델 예측과 후보의 합집합에서 불일치 픽셀에 `0.75 × 확신도 + 0.25 × 엔트로피`를 적용한 평균입니다. 가중치 없는 불일치와 함께 비교해 순위 변화도 확인합니다. AP, AUROC, 주입 유형별 상위 20% 발견 건수와 심각도 지표를 저장합니다. 같은 영상이 반복에 쓰이므로 반복 결과는 기술 통계로 해석합니다.

### 통제된 개선 실험

기준 모델의 검증 분석셋에서 작은 용종군과 복잡한 경계군을 각각 나머지 사례와 비교합니다. 하위군과 비교군에 최소 8장이 있고 Dice 격차가 더 큰 쪽을 택합니다. 동점이면 작은 용종 가설을 선택합니다. 적격 비교가 없으면 경계 가설을 탐색하며, 양의 격차가 관찰됐는지 보고서에 표시합니다.

- **작은 용종 가설**은 작은 용종 전경 픽셀의 BCE 가중치를 4배로 높이면 해당 그룹의 Recall이 개선된다는 가설입니다.
- **복잡한 경계 가설**은 참조 윤곽 주변 픽셀의 BCE 가중치를 4배로 높이면 해당 그룹의 Boundary F1이 개선된다는 가설입니다.

두 모델은 동일한 초기 가중치, 배치 순서, 데이터 증강, 학습 기간, 최적화 설정을 공유합니다. 자동 선택한 가설 하나만 학습하며, BCE 픽셀 가중치만 바꿉니다. 테스트 주 지표는 전체 macro Dice의 모델 간 차이입니다. 동일 이미지 예측을 짝짓고 이미지 중복 그룹을 2,000번 재표집해 95% percentile bootstrap 구간을 계산합니다. 이 구간은 한 분할·한 시드로 훈련한 두 모델의 테스트 표본 변동을 표시합니다.

## 산출물

| 파일 | 확인할 내용 |
| --- | --- |
| `RESULTS.md` | 실패 양상, 선택 근거, 검수 효율, 최종 성능과 해석 |
| `run_setup.json`, `split_manifest.csv`, `duplicate_candidates.csv` | 설정, 패키지·장치, 데이터 체크섬, 분할과 중복 후보 |
| `baseline_history.csv`, `intervention_history.csv` | epoch별 손실과 검증 지표 |
| `validation_failure_by_geometry.csv`, `improvement_hypothesis.json` | 형태별 성능과 가설 선정 근거 |
| `validation_*review_queue.csv`, `validation_*review_budgets.csv` | 정답 없는 검수 목록과 예측 실패 검수 예산 비교 |
| `validation_synthetic_annotation_*.csv` | 주입 결과, 순위, 예산별·유형별 탐지 성능 |
| `final_evaluation_plan.json`, `test_*.csv` | 고정 체크포인트 해시, 이미지별·짝지은 평가, 성능 비교표 |
| `figures/*.png` | 실패 사례, 학습 곡선, 형태별 실패, 검수 비교, 최종 테스트 |
| `checkpoints/*_best.pt`, `checkpoints/shared_initialization.pt` | 재사용 가능한 두 모델과 공통 초기 상태 |
| `checkpoints/*_last.pt` | 마지막 완료 epoch의 가중치, 옵티마이저, AMP와 학습 이력 |
| `kvasir_seg_results_<profile>_<id>.zip` | 결과 파일 모음. 기본값에서는 재개용 `*_last.pt`를 작업 경로에 따로 유지 |

## 재현과 복구

- **실행 환경.** seed 2026과 epoch별 시드로 셔플과 증강을 제어합니다. Colab의 CUDA 호환 torch/torchvision을 사용하고, 분석 패키지는 이미 설치된 버전을 사용하며 누락된 것은 설정 셀에 적힌 버전으로 설치합니다. Python, 패키지 버전, CUDA, GPU, 데이터와 분할 해시가 결과에 기록됩니다. CUDA 연산과 다른 하드웨어의 수치 차이가 남을 수 있습니다. [PyTorch 재현성 문서](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- **다운로드 실패.** 공식 배포본 ZIP을 Colab **파일** 탭에 업로드하고 첫 셀의 `LOCAL_ARCHIVE="/content/…zip"`을 지정합니다. 가중치 다운로드 실패 시 인터넷 연결과 TorchVision 버전을 확인한 뒤 다시 실행합니다.
- **런타임 초기화 또는 중단.** 같은 설정과 작업 경로로 모두 실행하면 `*_last.pt`가 있는 마지막 완료 epoch부터 재개합니다. 런타임이 사라진 뒤에도 이어가려면 첫 실행 전 `MOUNT_DRIVE=True`를 설정하고 인증합니다. 서로 다른 실행의 파일은 설정·데이터 해시에 따라 다른 경로에 저장합니다.
- **GPU 메모리 또는 워커 오류.** 런타임을 다시 연결하고, 메모리 오류가 발생하면 `BATCH_SIZE=4` 또는 `2`, 데이터 워커 오류가 발생하면 `NUM_WORKERS=0`으로 새 실행을 시작합니다.
- **재개용 체크포인트도 다운로드.** `INCLUDE_RESUME_CHECKPOINTS_IN_ZIP=True`로 설정한 뒤 마지막 셀을 실행합니다.

## 로컬 검증 상태

검증 상태는 구현 테스트를 설명합니다. 이 환경에는 CUDA GPU가 없어 Colab 기본 설정의 학습과 브라우저 ZIP 다운로드는 확인하지 못했습니다. Colab 실험 결과는 실제 실행 시 생성되는 `RESULTS.md`에서 확인합니다.

Python 3.12.14, torch 2.9.1+cpu, torchvision 0.24.1+cpu에서 노트북 형식과 코드 셀을 검사했습니다. 합성 데이터 48쌍으로 전체 셀을 실행해 산출물 생성을 확인했습니다. Simula URL이 이 환경에서 HTTP 503을 반환했으며, 원저자의 Kaggle v3 자동 대체 경로에서는 실제 영상과 마스크 1,000쌍을 내려받았습니다. TorchVision 공식 분할 가중치를 로드하고, CPU 96×96·배치 16·모델별 1 epoch 설정으로 두 모델을 각각 학습해 최종 ZIP까지 생성했습니다. 같은 설정으로 재실행했을 때 두 체크포인트가 재사용됐고 최종 평가 CSV가 일치했습니다.

축소 CPU 조건에서 검증셋의 20%를 검수했을 때 불확실성 순위는 예측 실패 16건을 발견했고, 무작위 반복의 평균은 14.08건이었습니다. 고정 테스트의 전체 Dice는 0.473에서 0.482로 변했고, 작은 용종군 Recall 차이는 -0.009였습니다. 이 실행에서는 작은 용종 Recall 향상 가설을 지지하지 않았습니다. 상세 수치와 생성 파일은 별도 CPU 검증 결과 ZIP의 `RESULTS.md`에서 확인할 수 있습니다.

## 연구 범위

20시간 설계 기준은 자료·실험 설계 3시간, 데이터·모델 구현 5시간, 실패·검수 실험 5시간, 통제 비교 4시간, 재현 검증·문서화 3시간입니다. 단일 공개 데이터셋과 단일 시드의 탐색 연구입니다. 모델이 확신을 갖고 틀리면 엔트로피가 낮을 수 있고, 합성 오류는 실제 주석 오류의 분포를 대변하지 않습니다. 임상 성능과 환자 안전성은 평가 범위에 포함되지 않습니다.

## 인용

Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. MultiMedia Modeling, 451–462. [논문](https://doi.org/10.1007/978-3-030-37734-2_37)
