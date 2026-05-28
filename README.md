<aside>

## 데이터셋 다운로드

### 데이터 분할 전

- ST4000DM000
    - 

---

### 데이터 분할 후

`(훈련:검증1:검증2:테스트 = 6:1:1:2)` 그룹 층화 분할

- train_raw.parquet (훈련)
    - 
- val_tune_raw.parquet (검증1)
    - 
- val_calib_raw.parquet (검증2)
    - 
- test_raw.parquet (테스트)
    - 

---

### 파생 변수 생성

train_raw.parquet으로부터 파생 변수를 생성한 후 8:2 그룹 층화 언더샘플링함 (클래스 불균형은 다름).

- fs_sample_train.parquet
    - 
- fs_sample_test.parquet

---

### 필터링 후 RFE 전용 데이터셋

group 내부 고상관 특성을 필터링한 결과물

- fs_train.parquet
    - 
- fs_test.parquet
    - 

---

### 변수 생성한 데이터셋

- train subset (train.parquet에서 42번 시드로 추출한 10개의 subset)
    - 
- val_tune.parquet (Reranking 용 원본)
    - 
- val_sampled.parquet (val_tune.parquet에서 샘플링된 Optuna 전용 검증셋)
    - 
- val_calib.qarquet
    - 
- test.parqeut
    - 

---

### 최종 모델

- 각 서브셋의 가중치 pkl파일
    - 
- 임계값
    
    
    | FPR | Recall | threshold |
    | --- | --- | --- |
    | 0.1% |  |  |
    | 0.5% |  |  |
    | 1.0% |  |  |
    | 5.0% |  |  |
</aside>

'max_depth': 9, 'num_leaves': 80, 'n_estimators': 506, 'learning_rate': 0.016534476178347253, 'min_child_samples': 49, 'feature_fraction': 0.6559834139907077, 'bagging_fraction': 0.9480114410775667, 'lambda_l1': 0.4111680887927762, 'lambda_l2': 0.01575019857369936

![image.png](attachment:b5046a6a-8f23-4221-a232-6277eed6cc66:image.png)

---

## **1. 데이터 정제**

연구 대상 단일 모델 추출 후 데이터 정리

- 모델 추출
    
    원본 데이터에서 `model`이 ST4000DM000인 모든 행을 추출함
    
    → ST4000DM000_raw.parquet로 저장
    
- 이후 아래의 공정을 거침
    
    → ST4000DM000_cleaned_1.parquet로 저장
    

```python
# 상수 열 (분산 0) 삭제 
# 정규화 열 삭제 (_raw 컬럼만 사용)
# 결측치 90% 이상 열 삭제
# 완전 중복 행 삭제
# 데이터 타입을 알맞게 수정
# 객체 별 타임라인으로 정렬
```

- 중복 날짜가 있는지 확인함. (없음)
- 디코딩 및 이상치 처리
    - [smart_1_raw](https://www.notion.so/smart_1_raw-33814366e96b800cb943c3f8df0aca0e?pvs=21) , ‣  디코딩
        
        <aside>
        
        Seagate 디스크의 SMART 1번과 7번 Raw 값은 '에러 횟수'와 '총 누적 작업량'이 48비트 패킹됨
        
        - **디코딩 연산 (비트 시프트)**
            
            10진수 원시값을 16진수로 변환 후, 비트 연산을 통해 상위 16비트(실제 에러 횟수, `>> 32`)와 하위 32비트(총 작업 횟수, `& 0xFFFFFFFF`)로 각각 분리 추출함.
            
        - **반영 결과**
            
            `smart_1_raw, smart_7_raw` 원본을 삭제하고, 비트 연산을 거친 변수 3개 `Total_Reads`, `total_seeks`, `seek_error_count`를 생성함.
            
        </aside>
        
    
    ```python
    # smart_1 삭제 → Read_Error_Count, Total_Reads
    # smart_7 삭제 → seek_error_count, total_seeks
    # - Read_Error_Count는 분산이 0이라 삭제함 -> 이거 좀 이상함. 이유 알아내야 함
    ```
    
    - [smart_188_raw](https://www.notion.so/smart_188_raw-33414366e96b80538a2ada639bd6f5f4?pvs=21)  디코딩
        
        <aside>
        
        188번 원시 값은 지연 심각도별 횟수(전체, 5초 이상, 7.5초 이상)가 16비트씩 3구간으로 쪼개져 압축되어 있음. 
        
        - **디코딩 연산 (비트 3단 분리)**
            
            원시 값을 비트 연산하여 3단계로 분해 추출함: `& 0xFFFF` (하위: 전체 초과), `>> 16 & 0xFFFF` (중간: 5초 지연), `>> 32 & 0xFFFF` (상위: 7.5초 지연).
            
        - **반영 결과**
            
            `smart_188_raw`를 삭제하고, 3가지 변수(`Timeout_Total`, `Timeout_5s`, `Timeout_7_5s`)로 생성함.
            
        </aside>
        
    
    ```python
    # smart_188 삭제 → Timeout_Total, Timeout_5s, Timeout_7_5s
    # Timeout_7_5s 삭제 (Timeout_5s와 극도로 높은 상관성, 응답 지연이 5초를 넘긴 것부터 매우 심각한 경우)
    ```
    
    - 분포 확인 및 이상치 처리
        
        EDA 결과 비정상적인 분포를 가진 컬럼에 대해 세부 분석
        
        <aside>
        
        [smart_12_raw](https://www.notion.so/smart_12_raw-33414366e96b807b9f5ce87ef10952cf?pvs=21) 
        
        - 문제점: 전원 온오프 횟수가 16억인 데이터 존재
        - 결론: 이상치 빼면 분산이 거의 0임. 불용
        
        ---
        
        [smart_189_raw](https://www.notion.so/smart_189_raw-33414366e96b807f8766dd3f7d2093e5?pvs=21) 
        
        - 문제점: 데이터 형식에 의해 최대값이 6만 부근에서 막힌 것으로 보여짐
        - 결론: 16비트 표현 한계인 `65535`에 도달한 뒤 더 이상 숫자가 올라가지 못하고 포화된 듯. 별다른 처리를 하지 않아도 될듯
        
        ---
        
        [smart_190_raw](https://www.notion.so/smart_190_raw-33414366e96b80528ac7df836a38fa64?pvs=21) 
        
        - 문제점: 온도의 최대값이 141도였음.
        - 결론: 시스템 오류임. 100도 이상인 값은 Foward fill로 덮어씌우기
        
        ---
        
        [smart_194_raw](https://www.notion.so/smart_194_raw-33414366e96b805db6fed24372a6ffb1?pvs=21) 
        
        - 문제점: 온도의 최대값이 141도였음.
        - 결론: 시스템 오류임. 100도 이상인 값은 Foward fill로 덮어씌우기
        
        ---
        
        [smart_197_raw ](https://www.notion.so/smart_197_raw-33414366e96b80f7964ffc72d8fc7ff0?pvs=21) 
        
        - 문제점: 데이터 형식에 의해 최대값이 6만 부근에서 막힌 것으로 보여짐
        - 결론: 문제 없음
        
        ---
        
        [smart_198_raw](https://www.notion.so/smart_198_raw-33414366e96b8067870ed67bc0a5c882?pvs=21) 
        
        - 문제점: 데이터 형식에 의해 최대값이 6만 부근에서 막힌 것으로 보여짐
        - 결론: 문제 없음
        </aside>
        
    
    ```python
    # smart_190_raw, smart_194_raw (디스크 온도)
    # 100도 이상인 값은 Forward fill로 덮어씌우기
    # smart_12_raw 삭제
    ```
    
- [smart_240_raw](https://www.notion.so/smart_240_raw-33414366e96b80ff8d8fc820426ae0ee?pvs=21)  삭제
    
    ```python
    # 디코딩 방식이 불명확하며, 헤드 실제 작동 시간은 smart_9_raw라는 대체재가 존재하여 삭제함
    ```
    
- 타겟변수 레이블링
    - 고장으로부터 10일 내의 구간`D-1 ~ D-30`을 1(고장 임박)로 레이블링
    - 고장 당일 `D-DAY` 삭제는 훈련 시에만 적용할 것임 (일단 아무것도 적용 안함)

<aside>

(필요)

- 원본 백블레이즈 데이터

(중간 산출물)

→ ST4000DM000_raw.parquet 

(산출물)

→ ST4000DM000_cleaned_1.parquet

</aside>

---

## 2. 데이터 정제2

- HDD 개체의 일 단위 기록 누락 처리
    
    ```python
    # - **Case 1)** 마지막이 1로 끝났다가 다시 기록되는 경우    
    #     → 해당 개체 데이터 전부 삭제
        
    # - **Case 2)** 마지막이 0으로 끝났다가 다시 기록되는 경우   
    #     → 1일 공백은 보간, 2일 이상 공백부터는 공백 전과 공백 후를 다른 시리얼 넘버를 부여. 
    				# (공백 후의 데이터들에 대해 기존 시리얼넘버 + _1씩 붙여주는 네이밍 규칙)    
    #     - 끊겨있던 구간이 포함되지 않게 파생변수를 생성하게 되는 효과 유도
    #     - 이후 데이터 분할 단계에서는 이를 같은 개체로 간주하여 분할함
    
    # - **Case 3)** 마지막이 0으로 끝난 경우 (이후 추가적인 데이터 없음)  
    #     → 마지막 30일 삭제 (라벨 오염 방지 조치)
    ```
    
- 사이에 빈 시계열 행 생성
- 일괄 Forward fill 적용
- 2014-03 이전 데이터 전부 제거 (~ 2014-02-마지막날)
- 여전히 결측치가 남아있다면 분기 조건으로 사용하도록 놔둠

→ ST4000DM000_cleaned_2.parquet로 저장

<aside>

(필요)

→ ST4000DM000_cleaned_1.parquet

(산출물)

→ ST4000DM000_cleaned_2.parquet

</aside>

---

## 3. 데이터 분할

그룹 층화 분할

- 개체를 구분하여 시간과 상관없이 랜덤 층화 분할 `훈련 : 검증 : 보정 : 테스트 = 6 : 1 : 1 : 2`
- 주의: 앞서 2일 이상의 공백은 시리얼번호_n 형식으로 분리했지만, 여기서는 하나의 개체로 취급해야 함
    - 물리적 개체 누수 방지 처리
        1. **물리적 동일성:** 이름표만 `A`와 `A_1`로 찢어졌을 뿐, 실제로는 공장에서 똑같이 찍혀 나온 **"완벽하게 똑같은 금속 하드디스크 1개"**입니다. 고유의 진동, 제조상 노이즈 등 물리적 특성(Signature)이 완전히 같습니다.
        2. **커닝(데이터 누수) 방지:** 만약 이 둘을 남남으로 취급하면, `A`는 수명이 짧아 Train 셋에 들어가고 나중에 부활한 `A_1`은 Test 셋에 들어가게 됩니다. 그러면 모델은 Train에서 이 하드디스크 특유의 물리적 패턴을 미리 외워버립니다(개체 누수).
        3. 결론: Test 셋은 반드시 '태어나서 한 번도 본 적 없는 낯선 디스크'만을 통과시켜야 합니다. 따라서 파생된 `_1`, `_2` 꼬리표들을 모두 원조 시리얼 넘버(Base) 기준으로 묶어 단일 그룹(Group)으로 간주하고, 무작위 층화 분할을 수행할 때 가족 단위 전체가 Train, Val, Test 중 한 곳으로 통째로 몰아져서 배정되도록(Family Binding) 처리하여 데이터 누수를 완벽히 차단합니다.
        

<aside>

(필요)

- ST4000DM000_cleaned_2.parquet

(산출물)

→ train_raw.parquet

→ val_tune_raw.parquet

→ val_calib_raw.parquet

→ test_raw.parquet

</aside>

---

## 4. 특성 생성

→ 각 fs 데이터셋 생성

자세한 내용은 [파생변수 생성](https://www.notion.so/34b14366e96b8051a73ec7aa7cbad935?pvs=21) 참고

- 모델의 고장 예측력을 극대화하기 위해, 원본 데이터를 기반으로 다양한 관점의 파생 변수를 생성
- 파생 변수 생성은 일률적인 통계적 특징 추출과 도메인 기반 특징 공학 두 가지 트랙으로 나누어 진행

<aside>

### 파생 변수 생성 전략

- 통계적 특징 추출
    - **설명:** 원본 데이터의 시계열적 변동성을 파악하기 위해 모든 주요 원본 및 차분 변수(오늘-어제)에 일괄 적용하는 통계적 기법입니다.
    - **윈도우 적용:** 3일, 7일, 14일, 28일 주기의 슬라이딩 윈도우(Sliding Window)를 적용합니다.
    - **추출 통계량:** 변수의 성격에 따라 불필요한 연산을 제외하고 `max`, `sum`, `평균`, `표준편차`, `z-score`, `asfd`, `dai`, `cid`, `accel`, `ewma` 등을 선별적으로 계산합니다.
- 도메인 기반 특징 공학
    - **설명:** 하드디스크의 물리적 동작 원리와 고장 메커니즘에 대한 도메인 지식을 바탕으로 직접 고안한 복합 파생 변수입니다.
    - **주요 지표:** 과거 물리적 손상 발생 이력(0/1 플래그), 시간당 작업량, 에러 발생 밀도 등 모델이 직접 학습하기 어려운 복합적인 패턴을 수치화합니다.

### 변수 명명 규칙

일관된 데이터 파이프라인 관리를 위해 아래의 명명 규칙을 엄격하게 적용합니다. (전체 소문자 및 언더바 연결)

- 통계적 특징 추출 **변수:** `[대상]_[기간]_[연산자]` 구조 (예: `s5_7d_mean`)
- 도메인 기반 특징 공학 **변수:** 여러 지표가 혼합되거나 특정 윈도우가 없는 경우, 변수의 직관적 물리적/논리적 의미를 담아 명명합니다. (예: `s187_14d_burst_index`)

### 데이터 결측 및 수치 예외 처리 규칙

데이터 변환 과정에서 발생할 수 있는 수치적 폭발과 정보 손실을 막기 위해 아래의 원칙을 적용했습니다.

**① ZeroDivisionError 방지 규칙**

분모가 0이 되는 상황을 방지하기 위해 변수의 데이터 타입에 따라 처리 방식을 이원화했습니다.

- **이산형 데이터 (빈도, 횟수, 개수 등): `+ 1` 처리 (라플라스 스무딩, Laplace Smoothing)**
    - *사유:* 미관측 범주에 대한 Zero-Probability 문제를 해결합니다. 이산형 데이터의 분모가 0일 때 아주 작은 엡실론을 더하면 결과값이 수십만 배로 폭발하여 트리 모델에 치명적인 스케일 붕괴(과적합)를 일으킬 수 있으므로 1을 더해 안정화합니다.
- **연속형 데이터 (비율, 표준편차 등): `+ ε` 처리 (수치적 안정화, Numerical Stabilization)**
    - *사유:* 미세한 소수점을 다루는 연속형 데이터에 상수 1을 더하면, 원래 데이터가 가진 예민한 변동성 신호가 훼손됩니다. 따라서 정보 손실을 최소화하기 위해 극소값 엡실론(1e-5)을 더해 연산 오류만 방지합니다.

**② 시계열 윈도우 기간 부족 (Cold Start) 처리**

- 차분 변수는 첫 행은 NULL로 둠.
- 윈도우 연산 시 과거 데이터가 부족한 구간도  `NULL`을 포함하여 계산
- 앞서 다수 결측치를 가진 90개 모델들도 마찬가지로 NULL 포함하여 계산

### 피처 생성 파이프라인 및 최종 산출물

- **차분 계산 → 통계적 특징 추출 윈도우별 계산 → 도메인 기반 특징 공학 계산** 순서로 진행
- 모든 산출물 파일은 데이터 압축 효율이 높은 `zstd` 포맷의 Parquet 형식으로 저장
- `serial_number`와 `date`를 복합 식별키로 사용
- 최종적으로 총 304 - 19개의 파생변수가 생성됨

| **파일명** | **포함 내역 및 설명** | **변수 개수** |
| --- | --- | --- |
| `fs_sample_diff.parquet` | **기초 데이터**
원본 데이터(19개) +  차분(18개) - failure - date/serial_number   | 40 - 3 = 37개 |
| `fs_sample_7d.parquet` | **7일 통계적 특징 추출 순수 통계량**
7일 윈도우가 적용된 핵심 속성(184, 194, 241, 242, Reads, Seeks, 190)의 통계량 | 58 - 2 = 56개 |
| `fs_sample_14d.parquet` | **14일 통계적 특징 추출 순수 통계량**
가장 많은 속성(17개)이 포함된 14일 주기 단기/중기 통계 데이터 | 76 - 2 = 74개 |
| `fs_sample_28d.parquet` | **28일 통계적 특징 추출 순수 통계량**
하드디스크의 장기 노화 상태 및 누적 피로도를 나타내는 28일 주기 데이터 | 76 - 2 = 74개 |
| `fs_sample_windowed.parquet` | **윈도우 기반 복합체**
윈도우형 도메인 기반 특징 공학(20개) | 18 - 2 = 16개 |
| `fs_sample_daily_status.parquet` | **디스크 상태 이력 (정적/이력)**
손상 여부 플래그 및 장애 발생 후 경과일 등 | 24 - 2 = 20개 |
| `fs_sample_daily_impact.parquet` | **디스크 부하 보고서 (동적/수치)**
에러 밀도, 작업량 비율 등 수치적으로 모델링된 부하 지표 | 30 - 2 = 27개 |
| **Total** | **RFE 투입 피처 총 개수 (중복 키 제외 순수 변수)** | **319 - 15 = 304개** |
</aside>

```markdown
# RFE용 데이터셋 제작
모든 고장 개체 사용, 정상 개체 샘플링 사용(모든 샘플링엔 seed=42 사용)
1. 고장 개체수 비율 fs_sample_train 8 : fs_sample_test 2 (홀드아웃)
2. 학습 및 테스트 세트 모두 정상 개체는 고장 개체수의 10배수 샘플링하여 배정
    - 원본 행 단위 불균형 1 : 1405.8이지만 타협한 수치
- (serial_number 단위로 움직이는 것)
```

<aside>

(필요)

- train_raw.parquet

(중간 산출물)

→ fs_sample_diff.parquet

→ fs_sample_7d.parquet

→ fs_sample_14d.parquet

→ fs_sample_28d.parquet

→ fs_sample_windowed.parquet

→ fs_sample_daily_status.parquet

→ fs_sample_daily_impact.parquet

(산출물)

→ fs_train.parquet

→ fs_validation.parquet

</aside>

---

## **5. 특성 선택**

- 재귀적 특성 제거(RFE, Recursive Feature Elimination)
    
    <aside>
    
    **[RFE(후진 제거 기법) 파이프라인 및 모델 운영 요약]**
    
    **1. RFE 파이프라인 동작 단계**
    
    - **초기 학습:** 전체 파생 변수를 포함하여 모델을 학습시킵니다.
    - **중요도 산출:** 학습된 모델에서 각 변수의 기여도(Feature Importance)를 추출하여 정렬합니다. (SHAP 변수중요도 가능)
    - **하위 변수 제거:** 기여도가 가장 낮은 변수를 제거합니다. (연산 효율을 위해 하위 2~3개를 그룹 단위로 제거할 수 있습니다.)
    - **성능 검증:** 남은 변수들로 교차 검증(CV)을 수행하여 목적 지표(PR-AUC 등)의 변화를 기록합니다.
    - **반복 수행:** 목표한 변수 개수에 도달하거나, 성능이 급격히 하락하는 임계점(Elbow Point)이 확인될 때까지 위 과정을 재귀적으로 반복합니다.
    
    **2. RFE 내부 LightGBM 파라미터 설정 전략**
    
    - **역할:** RFE 내의 모델은 최종 예측용이 아니라, 불필요한 변수를 신속하게 판별해 내는 도구입니다.
    - **설정 원칙:** RFE 과정에서만 수십 번의 반복 학습이 발생하므로 동적인 파라미터 튜닝은 배제합니다. 오직 **'연산 속도 확보'**와 **'클래스 불균형 제어'**에만 초점을 맞춘 **고정(Static) 파라미터**를 사용하는 것이 원칙입니다.
    
    ```python
    **권장 파라미터 세팅**
    
    - objective: 'binary' (이진 분류)
    - importance_type: 'gain' (★매우 중요)
    - class_weight 또는 scale_pos_weight: 'balanced' 또는 불균형 비율에 맞춘 값
    - max_depth: 5 ~ 7 (얕게 설정)
    - num_leaves: 31 ~ 50
    - n_estimators: 100 ~ 200
    - learning_rate: 0.05 ~ 0.1
    ```
    
    </aside>
    

<aside>

### 1. 특성 grouping

변수가 설명하는 의미 기반 그룹

- 그룹
    
    <aside>
    
    - 최근 부하 압력
    - 총 탐색량
    - 총 읽기량
    - 총 기록량
    - 누적 사용량
    - Burst 이상
    - Reallocated / Pending
    - 급성 Spike
    - Sector 열화
    - 읽기/쓰기 안정성
    - Seek 경로 이상
    - 기본 I/O 이상
    - 물리 스트레스 상호작용
    - 기계적 충격
    - 열 스트레스
    - 온도 수준
    - 시스템성 실패
    - 펌웨어 실패
    - 지속 악화
    - 최근 발생 시점
    - 최초 발생 시점
    - 과거 누적 발생
    - 직접 손상 발생
    </aside>
    

[변수 그룹 (json)](https://www.notion.so/json-35014366e96b80e58fb5ea57729c679f?pvs=21) 

### 2. group 내부 중복 변수 제거

- 그룹 내에서 상관계수가 0.9를 넘는 변수에 대해 가지치기
    - 규칙 + 도메인 지식 적용
    1. failure에서 더 뚜렷한 신호 (KS 통계량 기반)
        - KS 통계량
            
            
            ![image.png](attachment:00b44027-d48f-40d7-a6d7-be7db143e512:image.png)
            
            ![image.png](attachment:8e0e69e7-5601-4a5c-975d-3df83e8aae69:image.png)
            
            $KS = \max |F_{failure}(x) - F_{normal}(x)|$
            
            한 변수의 고장과 비고장 분포의 차이를 점수화
            
            두 분포를 각각 누적 분포 함수(CDF)로 변환하여 가장 먼 수직거리를 계산함.
            
            이 변수가 얼마나 고장을 뚜렷하게 변별하는가? 
            
    2. 해석 용이성 (고장을 설명하기 쉬운)
- 제거 변수
    
    ```json
        "drop_v1": [
            "error_saturation_score",
            "shock_seek_interaction",
            "log_shock_fly_interaction",
            "shock_fatigue_rate",
            "smart_191_raw",
            "s191_diff",
            "s189_diff",
            "s191_28d_sum",
            "s191_14d_sum",
            "s199_14d_max",
            "s199_diff",
            "s199_14d_burst",
            "s199_14d_sum",
            "s199_28d_sum",
            "s199_28d_max",
            "s191_14d_max",
            "s191_28d_max",
            "s194_diff",
            "s190_diff",
            "s189_28d_max",
            "s189_28d_highfly_burst",
            "s189_28d_sum",
            "s192_14d_burst",
            "temp_error_index",
            "is_warmup_7d",
            "total_reads_14d_sum",
            "total_reads_14d_mean"
        ]
        ---------------------
            "총 탐색량": [
            "total_seeks_7d_accel",
            "total_seeks_28d_accel",
            "total_seeks_7d_sum",
            "total_seeks_14d_sum",
            "total_seeks_28d_sum",
            "total_seeks_7d_mean",
            "total_seeks_28d_mean",
            "total_seeks_7d_ewma",
            "total_seeks_14d_ewma",
            "total_seeks_28d_ewma",
            "total_seeks_7d_std",
            "total_seeks_14d_std",
            "total_seeks_28d_std",
            "total_seeks_7d_asfd",
            "total_seeks_14d_asfd",
            "total_seeks_7d_max",
            "total_seeks_14d_max",
            "total_seeks_7d_zscore",
            "total_seeks_14d_zscore",
            "total_seeks_28d_zscore"
        ],
        "총 읽기량": [
            "s242_14d_asfd",
            "s242_14d_ewma",
            "s242_14d_max",
            "s242_14d_std",
            "s242_14d_sum",
            "s242_14d_zscore",
            "s242_28d_accel",
            "s242_28d_ewma",
            "s242_28d_max",
            "s242_28d_mean",
            "s242_28d_std",
            "s242_28d_sum",
            "s242_28d_zscore",
            "s242_7d_accel",
            "s242_7d_asfd",
            "s242_7d_ewma",
            "s242_7d_max",
            "s242_7d_mean",
            "s242_7d_std",
            "s242_7d_sum",
            "total_reads",
            "total_reads_14d_std",
            "total_reads_14d_zscore",
            "total_reads_28d_accel",
            "total_reads_28d_ewma",
            "total_reads_28d_sum",
            "total_reads_28d_zscore",
            "total_reads_7d_accel",
            "total_reads_7d_ewma",
            "total_reads_7d_mean",
            "total_reads_7d_sum",
            "total_reads_7d_zscore"
        ],
        "총 기록량": [
            "s241_7d_accel",
            "s241_28d_accel",
            "s241_7d_sum",
            "s241_14d_sum",
            "s241_28d_sum",
            "s241_7d_mean",
            "s241_28d_mean",
            "s241_7d_ewma",
            "s241_14d_ewma",
            "s241_28d_ewma",
            "s241_7d_std",
            "s241_14d_std",
            "s241_28d_std",
            "s241_14d_max",
            "s241_28d_max",
            "s241_7d_asfd",
            "s241_14d_asfd",
            "s241_14d_zscore",
            "s241_28d_zscore"
        ],
        "Reallocated / Pending": [
            "s5_14d_max",
            "s5_14d_sum",
            "s5_28d_sum"
        ],
        "Sector 열화": [
            "s198_error_rate",
            "s187_error_rate",
            "s197_28d_sum",
            "s198_28d_sum",
            "s197_28d_max",
            "s197_14d_sum",
            "s198_14d_sum",
            "s197_14d_max",
            "s198_14d_max",
            "s197_diff"
        ],
        "읽기/쓰기 안정성": [
            "s183_14d_sum",
            "s183_28d_sum"
        ],
        "Seek 경로 이상": [
            "seek_error_count_diff",
            "seek_error_14d_spike_ratio"
        ],
        "기본 I/O 이상": [
            "smart_199_raw",
            "timeout_total",
            "timeout_total_14d_sum",
            "timeout_total_diff",
            "timeout_total_28d_sum"
        ],
        "열 스트레스": [
            "s190_7d_asfd",
            "s190_7d_cid",
            "s190_14d_asfd",
            "s190_14d_cid",
            "s190_28d_asfd",
            "s190_28d_cid",
            "smart_190_raw"
        ],
        "온도 수준": [
            "s194_7d_asfd",
            "s194_7d_cid",
            "s194_14d_asfd",
            "s194_14d_cid",
            "s194_28d_asfd",
            "s194_28d_cid"
        ],
        "시스템성 실패": [
            "s184_3d_sum",
            "s184_7d_sum",
            "s184_14d_sum",
            "s184_diff"
        ],
        "직접 손상 발생": [
            "smart_5_raw",
            "s5_daily_failure_speed"
        ]
    ```
    

### 3. 특성 선택

- 진행중
</aside>

<aside>

(필요)

- fs_train.parquet
- fs_validation.parquet

(중간 산출물)

→ fs_train_filtered.parquet

→ fs_validation_filtered.parquet

(산출물)

→ train.parquet

→ val_tune.parquet

→ val_calib.parquet

→ test.parquet

</aside>

---

## **6. 하이퍼파라미터 최적화**

### 방법론

- 모델: LightGBM
- 방식: UnderBagging Ensemble (언더배깅 앙상블)
    
    <aside>
    
    ### 용어 설명
    
    - 배깅(Bagging): 붓스트랩(Bootstrap)으로 뽑아서 + 앙상블한다.
        
        → 붓스트랩은 복원 추출을 전제로 함. 우리는 비복원추출이니 엄밀한 의미의  배깅과는 차이가 있음.
        
        → 다만 용어 자체가 유명해서 비교적 느슨하게 사용되긴 함.
        
    - 언더배깅(Underbagging): 다수 클래스를 언더샘플링하여 여러 서브셋 구성 + 앙상블
        
        → 소수 클래스는 유지하고, 다수 클래스만 부분적으로 추출함
        
        → 극단적 불균형 환경에서 계산량을 줄이며 성능을 개선할 수 있음.
        
    </aside>
    

<aside>

### 사용된 데이터 설명

full train: 학습을 위한 10개의 서브셋을 만들고 더 이상 사용되지 않음.

- 10개의 Subset:  앙상블 학습을 위한 경량 데이터셋
    - 모든 고장 개체 (고정) + 같은 개수의 정상 개체 (서브셋 내 / 서브셋 간 비복원 추출)

val_tune.parquet: Optuna 루프에서는 미사용, 최종 선택 (rerank) 과정에서 사용됨

- val.sampled.parquet: 옵튜나 내부에서 사용되는 경량 데이터셋
    - val_tune.parquet 고장 개체의 80% 랜덤 추출 + 그에 상응하는 3배수로 추출된 정상 개체
</aside>

### [Optuna]

```markdown
1. 10개의 subset을 순차적으로 학습함 (롤링 추론, 고장 당일 포함 안함)

2. 각 subset 학습 직후:
    - val_sampled.parquet을 예측하고
    - 현재까지의 누적 ensemble PR-AUC 계산
    
3. 중간 PR-AUC를 Optuna에 리포트함:
    - 내부적으로 이전 trial 성능과 비교하여
    - 가지치기(pruning) 진행
    - 남은 subset 학습 중단하고 다음 trial 진행
    
4. pruning 되지 않은 경우
    → Optuna에 최종 ensemble PR-AUC 반환
    
5. Optuna(TPE)가 내부적으로 다음 하이퍼파라미터 후보를 생성함
    - 베이지안 최적화 기반
    - 이를 반복함
```

---

### [Reranking]

- val_sampled.parquet에 편향되는 리스크를 최소화하기 위한 재평가 단계
- full validation 기준 최종 선택 (전체 검증 데이터 사용)

```markdown
1. 앞의 결과 중 best parameter이 될 수 있는 후보 선택
	- 분포를 보고 n개의 trial을 선정함

2. 저장된 ensemble 모델 재사용

3. val.tune을 예측하여 PR-AUC 계산

4. 이를 기준으로 최종 하이퍼파라미터 선택
```

- 탐색 공간 (수정 필요)
    
    
    | **Parameter** | **Type** | **Range** | **Log Scale** | **Rationale (설계 논리)** |
    | --- | --- | --- | --- | --- |
    | **`learning_rate`**  | Float | `0.01 ~ 0.1` | **Yes** | 트리 개수(`n_estimators=400`)가 고정된 상태에서 수렴 속도를 맞추기 위해 넓은 로그 스케일로 탐색합니다. |
    | **`max_depth`** | Int | `4 ~ 10` | No | 과적합 방지를 위한 1차 방어선입니다. SMART 데이터의 노이즈 특성상 10을 초과하는 깊이는 유의미한 패턴보다 노이즈를 외울 확률이 높습니다. |
    | **`num_leaves`** | Int | `16 ~ 128` | No | 과적합 방지를 위한 2차 방어선입니다. 무의미한 탐색(예: depth=4인데 leaves=100)을 막기 위해, 실제 코드에서는 `min(2^max_depth, 128)`로 **동적 상한(Conditional Bound)**을 걸어 공간 낭비를 제거했습니다. |
    | **`min_child_samples`** | Int | `20 ~ 100` | No | 불균형 데이터에서 리프 노드가 극소수의 Positive 샘플에 과적합되는 것을 막는 최소한의 허들입니다. |
    | **`feature_fraction`** | Float | `0.6 ~ 1.0` | No | **(핵심)** Correlated SMART Feature 환경에서 트리가 특정 강한 노이즈 피처에 종속되는 것을 막고, 각 트리의 Subspace Sampling을 강화하여 **앙상블(Tree-level) 다양성을 증가시킴**. 단, 부스팅 과정 파괴를 막기 위해 마지노선(0.5)을 방어합니다. |
    | **`bagging_fraction`** | Float | 1 | No | 이미 외부에서 10:1 Underbagging 앙상블이 적용되어 있으므로, 내부의 Row-sampling은 중간 수준(0.6) 이상으로 유지하여 학습 데이터 손실을 막습니다. |
    | **`lambda_l1`** | Float | `0` | **Yes** | L1 규제. 불필요한 split 사용을 억제하여 noisy feature 의존을 완화 |
    | **`lambda_l2`** | Float | `0` | **Yes** | L2 규제. 잎사귀(Leaf)의 출력값을 부드럽게 눌러주어 leaf output의 과도한 진폭을 완화하는 방향으로 작용 |
    | **`bagging_freq`** | Int | **`1` (고정)** | - | Underbagging 체제이므로 항상 bagging이 켜져 있는 상태를 유지합니다. |
    | **`n_estimators`** | Int | **`400 ~ 800`** | - | Stage 1 에서는 빠른 파라미터 영역 필터링(Coarse Screening)과 파라미터 간 n_estimators 변동에 따른 탐색 노이즈를 줄이기 위해 고정 |
    | scale_pos_weight |  | 1 |  |  |

---

개체 단위 롤링 추론 방식을 기준으로 한 하이퍼파라미터 튜닝

|  | 이번 결과 | 행 단위 추론(기존 결과) | 베이스라인 |
| --- | --- | --- | --- |
| 개체 단위 롤링 추론 기준 PR-AUC |  | ? | ? |
| 행 단위 추론 PR-AUC | ? | 0.12721 | 0.111309(언더배깅 적용), 0.096265 |

<aside>

(필요)

- seed_42
    - subset_0~9.parquet
- val_tune.parquet

(중간 산출물)

→ val_sampled.parquet

(산출물)

→ models\underbagging_ensemble

- subset_00~09.pkl
</aside>

옵튜나 연산 효율성을 위한 학습 전략

1. 학습 데이터 언더샘플링하여 저장
    
    모든 고장 데이터 (고정) + 같은 개수의 정상 데이터 랜덤 샘플링하여 10개의 서브셋 생성
    
    서브셋 내 / 서브셋 간 비복원 추출 보장
    
    → 이후 언더배깅 시행
    
2. 언더샘플링한 하나의 검증 데이터셋 생성
    
    고장 개체의 80%만 추출하여 정상 개체를 10배수 붙임. 
    
    → 이것으로 옵튜나 내부 루프를 돌음
    
3. 이후 옵튜나 상위 결과들로 각각 전체 검증 데이터을 예측하여 리랭킹을 함.

---

## 7. 임계값 튜닝

제약 조건 최적화

<aside>

개체 단위 롤링 평가 적용

1. val_calib.parquet로 임계점 그리드서치 (1000등분)
2. FPR과 Recall 의 관계를 그래프로 그림 (x: FPR, y: Recall(TPR))
3. 적절한 지점을 도출함

→ 허용한 오탐율 내에서 탐지 성능을 최대화한 운영점

</aside>

## disk-level threshold

롤링 추론 결과 (리드타임 = 30일)

평가 기준은 리드타임 내에서 생애 최초 알람이 울려야만 고장 탐지 성공으로 간주

| 허용 오탐율 | threshold | Disk FAR | Disk Recall | Precision |
| --- | --- | --- | --- | --- |
| 0.10   %  |  |  |  |  |
| 0.50   %  |  |  |  |  |
| 1.00   %  |  |  |  |  |
| 2.00   %  |  |  |  |  |

<aside>

(필요)

- 모델 가중치 pkl
- val_calib.parquet

(산출)

→ threshold 기준

</aside>

---

## 8. 모델 평가

## **실무형 롤링 평가**

- 설명
    
    실제 운영 환경을 모사하여 시간 축과 개체 단위를 반영한 평가를 수행한다.
    
    - **평가 방식**
        - 1일 단위 rolling inference (sliding window)
        - 시계열 순서를 유지한 예측 수행
    - **평가 단위**
        - 개체(Entity: serial number) 기준 평가
    - **핵심 지표**
        - 고장 탐지율 (Hit Rate)
        - 평균 사전 경보 시간 (Lead Time)
            - 탐지 못한 개체는 계산에서 제외
        - 오탐율 (정상 개체에서 잘못 울린 비율)
        - 미탐율 (고장 개체에서 안울린 비율)
    - **특징**
        - 실제 운영 환경과 동일한 구조

<aside>

- 임계값 별 성능
    
    
    | Threshold | Disk FAR | Disk Recall | Precision |
    | --- | --- | --- | --- |
    | 0.9820 |  |  |  |
    | 0.9720 |  |  |  |
    | 0.9590 (메인) |  |  |  |
    | 0.9401 |  |  |  |
- 리드타임 별 성능
</aside>

<aside>

(필요)

- 모델 가중치 pkl
- test.parquet

(산출물)

→ 성능 지표

→ 리드타임 분석 결과

</aside>

---

## 9. 고장 해석 (수정 전)

### **전역적 해석 (Global Interpretability)**

- SHAP
    - 에러와 관련된 급성 지표 원본이 상위권을 차지함.

---

### **국소적 시간 기반 해석 (Temporal Interpretation)**

랜덤으로 표본을 추출하여 국소적/시간 기반 분석함.

### 정탐 (고장 개체를 리드타임 내에 탐지함)

### 오탐 (정상 개체에 알람을 잘못 울림)

### 오탐 (고장 개체를 리드타임 밖에서 탐지함)

### 미탐 (고장 개체에 알람을 울리지 않음) ← Recall 붕괴 원인

---

## 한계점 (수정 전)

1. 모델이 잡아내지 못한 미탐 사례가 많음.
    - 
2. 오탐율 대비 재현성이 올라오는 시점이 다소 느림
    - 
3. 분류 성능 지표의 한계
    - 
4. 단일 모델 / 단일 제조사 제한
    - 
5. 미래 누수에 대한 가능성
    -