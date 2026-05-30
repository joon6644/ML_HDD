# HDD 고장 예측 파이프라인 절대 기준 (Pipeline Standards)

이 문서는 프로젝트의 데이터 분할, 모델 학습, 임계값 튜닝 및 최종 평가 과정에서 반드시 준수해야 하는 **절대적인 설계 및 개발 기준**을 정의합니다.

---

## 1. 평가 기준 (Evaluation Standards)
* **개체(Drive) 단위 평가 통일**:
  * 모델 평가는 단순히 행(Row) 단위의 오차 행렬(Confusion Matrix)에 의존하지 않으며, 무조건 **디스크 개체(Entity, `serial_number`) 단위**로 수행합니다.
  * 개체 식별 시 임시로 생성된 접미사(예: `_n` 등)는 무시하고, 원본 고유 시리얼 번호로 통일하여 매칭하고 평가해야 합니다.
* **롤링 추론 (Rolling Inference)**:
  * 슬라이딩 윈도우(Sliding Window)를 사용해 연속된 시계열 상에서 고장 시점을 예측하며, 특정 윈도우 내 최초 경보를 기준으로 탐지 여부를 판단합니다.
* **30일 고장 예측 및 외부 종속성 제거**:
  * 고장 예측 리드타임(Lead Time)은 **30일**을 기준으로 삼습니다.
  * 평가 코드 및 성능 지표 계산 시 레거시 시스템이나 외부 라이브러리에 대한 종속성을 철저히 배제하고, 독자적이고 재현 가능한 로직(`src/eval_core.py` 등)을 이용해 연산합니다.

---

## 2. 경로 및 저장 기준 (Directory & Storage Standards)
* **접미사 `2` 경로 사용 규칙**:
  * 기존 실험 및 파이프라인과의 혼선을 방지하기 위해, 신규 파이프라인에서 생성되는 모든 데이터, 모델, 코드 등은 폴더명 끝에 **`2`**를 붙여 저장합니다.
    * **노트북**: `C:\Workspace\06_ML_projdect\26_1_COIN\notebooks2\`
    * **데이터셋**: `C:\Workspace\06_ML_projdect\26_1_COIN\data2\`
    * **모델 및 가중치**: `C:\Workspace\06_ML_projdect\26_1_COIN\notebooks2\models2\`
* **모델 가중치 및 설정 저장**:
  * 학습 및 튜닝 완료 시 산출물은 반드시 `notebooks2/models2/06d_optuna_tuning/seed_42` 하위에 저장하고 이곳으로부터 로드해야 합니다.
  * 하드코딩된 경로(예: 기존 `models/underbagging_ensemble_4`)를 연결하지 마십시오.

---

## 3. 학습 및 설정 기준 (Training & Configuration Standards)
* **피처 리스트 준수**:
  * `06a_feature_engineering.ipynb`에서 확정된 최종 변수 리스트를 피처 엔지니어링 및 모델 학습 시 누락 없이 정확하게 반영해야 합니다.
* **동적 서브셋 감지 (Dynamic Subset Detection)**:
  * 서브셋의 개수(`N_SUBSETS`)를 임의로 하드코딩하지 않습니다.
  * `data2/06_subset_generation/seed_42` 디렉토리에 물리적으로 분할된 parquet 파일 개수를 스캔하여 동적으로 설정값을 적용하도록 구현을 유지해야 합니다.
* **안전한 학습 로그 보존**:
  * Optuna 최적화 수행 시, 학습 과정에서 오류나 중단이 발생하더라도 진행 상황이 유실되지 않도록 매 트라이얼(Trial)마다 `notebooks2/optuna_trials.csv`에 진행 이력을 실시간 콜백으로 기록합니다.
  * 학습 DB(`optuna_study.db`) 역시 `notebooks2/` 하위에 두고 중단 시 바로 이어서 학습 가능한 상태로 관리합니다.

---

## 4. 산출물 및 리포트 저장 기준 (Output & Report Storage Standards)
* **결과물 분리 및 정리**:
  * 모든 시각화 차트 이미지(PNG 등), 실험 로그(CSV), 기타 분석 엑셀 파일 등은 노트북 루트 디렉토리에 흩뿌리지 않고 지정된 하위 폴더에 체계적으로 저장해야 합니다.
* **지정된 저장 폴더**:
  * **학습 로그 및 하이퍼파라미터**: `notebooks2/models2/06d_optuna_tuning/seed_42/` 하위에 저장 (예: `best_params.json`, `feature_cols.json`, `best_threshold.json` 등).
  * **임계값 튜닝 및 평가 보고서**: `notebooks2/reports2/` 하위 경로에 저장 (예: `fpr_recall_table.csv`, `threshold_grid.csv` 등).
  * **시각화 플롯 (이미지)**: `notebooks2/reports2/plots/` 하위에 파일로 저장하여 노트북 외부에서도 확인 및 리포트 작성이 가능하도록 관리합니다.
  * **노트북 루트 깨끗이 유지**: 소스 코드인 `.ipynb` 파일이 있는 루트 폴더는 오직 실행 소스만 두고, 자동 생성되는 이미지나 결과 엑셀 등으로 지저분해지지 않도록 청결히 유지합니다.
* **노트북 번호 접두사(Prefix) 규칙**:
  * 생성되는 모든 결과물 파일(CSV 테이블, PNG 시각화 이미지 등)의 파일명 맨 앞에는 그것을 생성한 노트북의 번호(예: `07a_`, `07b_`, `08b_`, `08c_` 등)를 접두사로 반드시 붙여 저장함으로써 유래를 한눈에 식별할 수 있도록 합니다.

