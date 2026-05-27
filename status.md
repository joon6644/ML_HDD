# HDD 고장 예측 프로젝트 현황 요약 (status.md)

이 문서에는 현재 수행 중인 모델 학습 및 평가 분석의 진행 상황, 핵심 쟁점(데이터 누수 및 평가 단위 불일치), 향후 계획, 참고 코드가 정리되어 있습니다.

---

## 1. 지금 하고 있는 것
* **베이스라인 모델 구축 및 최종 모델과의 성능 비교**
  * 원본 데이터(`ST4000DM000_v3.parquet`)의 24개 원본 SMART 변수만 사용하여 모델을 학습하고 임계값 최적화 및 롤링 추론을 진행하는 베이스라인 모델(방법A: No-Tricks / 방법B: UnderBagging)을 구축했습니다.
  * 최종 엔지니어링 모델(27개 파생 변수 및 Optuna 하이퍼파라미터 튜닝)과의 공정한 비교를 위해 고장 당일(D-DAY) 데이터를 추가하는 파이프라인(`10a_baseline_data_split.ipynb`)을 동일하게 구현해 평가했습니다.

---

## 2. 문제가 되는 것 (핵심 분석 결과)

### ① 평가 그룹화 단위(Grouping Unit)의 결정적 차이 및 왜곡
* **현상**: 베이스라인 노트북(`10b_baseline_training.ipynb`)에서는 테스트 셋 성능이 **FAR 0.74%, Recall_30d 33.72%**로 높게 보고되었으나, 최종 모델 성능 리포트(`08c_final_evaluation_labeled.ipynb`)에서는 **FAR 1.20%, Recall_30d 23.07%**로 현저히 낮게 기록되었습니다.
* **원인**: **평가 시 디스크를 그룹화하는 기준(`serial_number` vs `base_serial`)이 서로 달랐습니다.**
  * **베이스라인 노트북 (`10b`)**: `serial_number` 기준 그룹화
    * v3.parquet에는 중간 날짜 공백(Case 2)으로 인해 하나의 물리 디스크가 `S300VL64_0`, `S300VL64_1` 등 여러 서브세그먼트로 쪼개져 있습니다.
    * 이를 개별 디스크로 취급하면, 고장나지 않은 앞부분의 세그먼트(`_0`)들이 모두 **독립된 정상 디스크**로 평가에 포함되어 분모(정상 디스크 수)가 늘어나고, 결과적으로 **FAR(오탐율)이 인위적으로 낮게 계산(0.74%)**됩니다.
  * **최종 모델 평가 (`08c`)**: `base_serial` 기준 그룹화
    * 접미사(`_\d+$`)를 제거하여 하나의 물리 디스크 단위로 생애주기를 통합하여 평가합니다.
    * 베이스라인 모델(방법 B)을 최종 모델과 동일하게 `base_serial` 기준으로 재평가(실험 완료)하면 다음과 같이 변화합니다.
      * 기존 `serial_number` 기준: **FAR 0.74% | Recall_30d 33.72%** (임계값 0.8328)
      * 수정 `base_serial` 기준: **FAR 1.50% | Recall_30d 32.92%** (임계값 0.8328)
      * 동일 FAR 제약(FDR <= 1.0%) 내 재정렬 시: **FAR 0.90% | Recall_30d 29.57%** (임계값 0.9000)

### ② 최종 모델의 성능 하락 (베이스라인 대비 열세)
* 동일한 `base_serial` 기준으로 비교하더라도, 27개 파생 변수와 Optuna를 적용한 최종 모델(Recall_30d 23.07% @ FAR 1.20%)이 오히려 베이스라인 모델(Recall_30d 29.57% @ FAR 0.90%)보다 성능이 낮게 나타납니다.
* **PR-AUC 분석**:
  * **베이스라인 (방법 B)**: 행(Row) 단위 PR-AUC = **0.096265**
  * **최종 앙상블 모델**: 행(Row) 단위 PR-AUC = **0.111309**
  * 행 단위 예측 품질(PR-AUC)은 최종 모델이 우수하지만, 디스크 단위의 롤링 알람 평가 시에는 최종 모델의 오탐이 더 잦아 실질적인 Recall-FAR 트레이드오프에서 열세를 보입니다.

### ③ 누적성(Cumulative) 변수의 영향성 확인
* 베이스라인에 포함된 누적형 원본 변수(`total_seeks`, `smart_9_raw` 등)들의 데이터 누수 혹은 영향성을 평가하기 위해, 이를 제외한 19개 변수로 베이스라인을 재학습했습니다.
* **제외 결과** (`base_serial` 기준 평가):
  * 누적 변수 포함: **FAR 0.90% | Recall_30d 29.57%** (임계값 0.9000)
  * 누적 변수 제외: **FAR 0.99% | Recall_30d 26.83%** (임계값 0.8768)
  * 누적 변수가 제외되면서 Recall이 약 2.74%p 하락하였으나, 최종 모델(23.07%) 대비 여전히 높은 성능을 보입니다. 즉, 성능 격차가 단순히 누적 변수의 유무 때문만은 아님을 시사합니다.

---

## 3. 분석 결과 및 조치 완료 내용

### 1) 평가 기준의 단일화 (base_serial 통일) - 조치 완료
- 베이스라인 평가 코드(`10b_baseline_training.ipynb`)를 `base_serial` 기준으로 전면 수정하였습니다.
  - **`predict_rolling`** 함수 내에서 디스크를 `base_serial` 기준으로 정렬하도록 수정하였습니다.
  - **`prepare_eval_data`** 함수 내에서 디스크를 `base_serial` 기준으로 그룹화하도록 변경하여, 물리 디스크 조각(`_0`, `_1` 등)이 시간 순서대로 정렬 및 통합되어 평가되도록 하였습니다.
  - **`build_subsets`** (방법 B 언더배깅 서브셋 생성) 함수에서도 가중치 부여 대상 고장 날짜(fail_date) 산출 시 `base_serial` 기준으로 그룹화하여 매칭하도록 보완하였습니다.
  - 이에 따라 베이스라인 모델 역시 최종 모델과 동일하게 `base_serial` 관점에서 공정하게 비교할 수 있는 환경이 조성되었습니다. (최종 실행은 사용자가 수행 예정)

### 2) 최종 모델의 오탐 원인 분석 - 분석 완료
- 정상 디스크(Normal Disks) 6,255개 중 조기 경보(False Alarm)를 유발한 82개(1.31%) 디스크의 알람 시점 데이터를 분석하였습니다.
- **주요 오탐 유발 피처 (Standardized Difference 및 SHAP 기여도 기준):**
  1. **`smart_198_raw`** (Offline Uncorrectable Sector Count) - SHAP: 2.0222, Std Diff: 40.0699
  2. **`error_density_14d`** (14일 평균 에러 밀도) - SHAP: 2.0028, Std Diff: 27.9146
  3. **`smart_187_raw`** (Reported Uncorrectable Errors) - SHAP: 0.6698, Std Diff: 14.6036
  4. **`multi_error_count`** (다중 에러 카운트) - SHAP: 0.5027, Std Diff: 71.1565
- **오탐 원인 해석:**
  정상 디스크 중 일부가 일시적인 배드 섹터나 단순 언디텍티드 오류 등으로 인해 `smart_198_raw`나 `smart_187_raw` 값이 튀는 현상이 발생합니다. 모델은 이 피처들을 고장의 강력한 징후로 판단하여 예측 확률을 대폭 상승시킵니다. 이로 인해 정상 디스크에서 오탐이 빈번하게 발생하였고, 허용 False Alarm Rate(FAR <= 1%) 제약조건을 만족하기 위해 임계값이 극단적으로 높게(T = 0.9720 또는 0.9261) 설정될 수밖에 없었습니다. 높은 임계값은 결과적으로 실제 고장 디스크들의 탐지 시점을 늦추거나 미탐(FN)으로 만들어 전체 Recall을 떨어뜨리는 결과를 낳았습니다.

### 3) 피처 엔지니어링 및 하이퍼파라미터 튜닝 파이프라인 재점검 (과적합 검토) - 분석 완료
- **학습과 평가의 목적 함수 괴리:**
  - **학습/튜닝 단계 (`val_tune`):** Optuna를 사용해 샘플링된 검증 데이터셋(`val_tune_sampled.parquet`)에서 **행(Row) 단위 PR-AUC**를 극대화하도록 파라미터를 튜닝하였습니다.
  - **평가 단계:** 슬라이딩 윈도우(14일) 내에서 특정 횟수(2회) 이상의 알람이 발생하면 디스크 전체를 고장으로 판정하는 **디스크(Disk) 단위 롤링 알람 FAR & Recall**을 사용합니다.
- **과적합 및 괴리 메커니즘:**
  - Optuna는 행 단위 분류 정확도(PR-AUC)를 높이기 위해 에러가 발생한 극소수의 행들을 확실하게 고장(1)으로 밀어내도록 파라미터를 학습시켰습니다.
  - 이 과정에서 에러 밀도나 섹터 에러 피처에 과도한 가중치를 부여하게 되어, 실제로는 고장 나지 않는 정상 디스크의 에러 발생 행들마저 예측 확률이 극단적으로 치솟게 되었습니다.
  - 행 단위 PR-AUC는 향상되었을지 모르나, 디스크 단위 평가에서는 단 1회의 예측 확률 급증(혹은 14일 내 2회)으로도 디스크 전체가 False Alarm으로 간주되므로, 디스크 레벨 FAR이 폭발적으로 증가하였습니다.
  - 결국 이를 억제하기 위해 임계값을 상향하면서 최종 디스크 레벨 Recall(23.07%)이 튜닝을 거치지 않은 베이스라인(29.57%)보다 낮아지는 현상이 발생하였습니다.

---

## 4. 참고할 것

* **최종 모델 평가 데이터 및 예측 확률 캐시 경로**:
  * 데이터: `C:\Workspace\06_ML_projdect\26_1_COIN\data\split_group_stratified\test_with_failure_date.parquet`
  * 예측 확률: `C:\Workspace\06_ML_projdect\26_1_COIN\models\underbagging_ensemble_4\test_with_failure_date_probs.npy`
* **분석 결과 파일**:
  * 정상 디스크 오탐 상세 통계 및 SHAP 보고서: [scratch/false_alarm_analysis_report.md](file:///C:/Workspace/06_ML_projdect/26_1_COIN/scratch/false_alarm_analysis_report.md)
* **핵심 참조 노트북**:
  * 베이스라인 데이터 준비: [10a_baseline_data_split.ipynb](file:///C:/Workspace/06_ML_projdect/26_1_COIN/notebooks/10a_baseline_data_split.ipynb)
  * 베이스라인 학습/평가: [10b_baseline_training.ipynb](file:///C:/Workspace/06_ML_projdect/26_1_COIN/notebooks/10b_baseline_training.ipynb)
  * 최종 평가: [08c_final_evaluation_labeled.ipynb](file:///C:/Workspace/06_ML_projdect/26_1_COIN/notebooks/08c_final_evaluation_labeled.ipynb)
  * 롤링 평가 기반 Optuna 최적화: [11a_rolling_optuna_tuning.ipynb](file:///C:/Workspace/06_ML_projdect/26_1_COIN/notebooks/11_rolling_optuna/11a_rolling_optuna_tuning.ipynb)
* **임시 검증용 스크립트 (Scratch)**:
  * 최적 임계값 및 FAR-Recall 곡선 초고속 검증: [scratch/fast_eval.py](file:///C:/Workspace/06_ML_projdect/26_1_COIN/scratch/fast_eval.py)
  * 누적 변수 제외 베이스라인 학습 스크립트: `C:\Workspace\06_ML_projdect\26_1_COIN\train_no_cumulative.py`

