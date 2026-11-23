# 내시경 용종 분할과 마스크 검수

Colab에서 Kvasir-SEG 내시경 영상으로 용종 분할 모델을 학습하고, 마스크가 어디서 어긋나는지 살펴봤습니다. 경계가 복잡한 영상에서 오류가 잦았습니다. 모델이 불확실해하는 영상부터 고르면 같은 검수 건수로 실패를 더 많이 찾을 수 있는지 확인해 봤습니다.

검증 영상 100장 중 20장을 골랐을 때, 불확실성이 높은 순서로는 실패 4건을 찾았고 무작위로는 평균 2.35건을 찾았습니다. 이 순서를 검수 화면에 적용해 마스크를 바로 고치고 판단 기록을 남길 수 있게 했습니다.

## Colab 실행 순서

1. [모델 학습과 오류 분석](notebooks/Kvasir_SEG_Research.ipynb) · 학습한 `baseline_best.pt`를 저장합니다. [실행 안내](docs/research.md)
2. [마스크 검수](notebooks/Kvasir_SEG_Review_Colab.ipynb) · 마스크를 승인·수정·보류하고 `review_results.zip`을 받습니다. [실행 안내](docs/review.md)
3. [검수 전후 마스크로 재학습](notebooks/Kvasir_SEG_Reviewed_Label_Finetune.ipynb) · 검수 ZIP과 체크포인트로 두 조건을 비교합니다. 이 단계는 합성 데이터로 실행을 확인했습니다. [실행 안내](docs/finetune.md)

사용 데이터 · [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) · [Jha et al., 2020](https://doi.org/10.1007/978-3-030-37734-2_37)
