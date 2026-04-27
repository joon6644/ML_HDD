<aside>

## 데이터셋 다운로드

### 데이터 분할 전

- ST4000DM000_v3.parquet
    - https://drive.google.com/file/d/1otLyzULzpUBAG22Ayl3LAF9Zrx2U3Zoy/view?usp=sharing
    - EDA 결과
        
        ```python
        [C:/Workspace/06_ML_projdect/26_1_COIN/data/ST4000DM000_v3.parquet]
        종합 EDA 및 데이터 무결성 검증을 시작합니다 ...
        
        === [1. 데이터 규격 확인] ===
        총 행 수(Rows): 79,698,388 개
        총 열 수(Columns): 27 개
        
        === [2. 상위 5개 데이터 샘플 (생략 없음)] ===
          serial_number       date  smart_3_raw  smart_4_raw  smart_5_raw  smart_9_raw  smart_10_raw  smart_183_raw  smart_184_raw  smart_187_raw  smart_189_raw  smart_191_raw  smart_192_raw  smart_193_raw  smart_197_raw  smart_198_raw  smart_199_raw  smart_241_raw  smart_242_raw  Total_Reads  seek_error_count  Total_Seeks  failure  timeout_total  timeout_5s  smart_190_raw  smart_194_raw
        0    S3008532_1 2014-04-19            0            2            0          562             0              1              0              0              1              0              1           1541              0              0              0     2721830352     8918128525    101001152                 0     29220189        0              0           0             19             19
        1    S3008532_1 2014-04-20            0            2            0          586             0              1              0              0              1              0              1           1541              0              0              0     2725742872     9298167283     47333632                 0     31344455        0              0           0             19             19
        2    S3008532_1 2014-04-21            0            2            0          610             0              1              0              0              1              0              1           1541              0              0              0     2727616876     9383480381     64054464                 0     32087252        0              0           0             19             19
        3    S3008532_1 2014-04-22            0            2            0          634             0              1              0              0              1              0              1           1543              0              0              0     2748947918     9627727670     64019112                 0     33729684        0              0           0             19             19
        4    S3008532_1 2014-04-23            0            2            0          658             0              1              0              0              1              0              1           1543              0              0              0     3138614586     9750647272    206676600                 0     37215343        0              0           0             19             19
        
        === [3. 하드디스크 개체 및 클래스 분포 통계] ===
        [개체 단위 통계 (물리적인 하드디스크 개수)]
        - 전체 고유 개체 수 (접미사 통합): 37,024 개
        - 정상 작동 하드: 31,282 개 (84.49%)
        - 고장 발생 개체: 5,742 개 (15.51%)
        - 개체 단위 비율 (Class 1 : 0) = 1 : 5.45
        
        [행 단위 클래스 분포 (❗진짜 ML 모델이 학습할 Target 레이블 비율)]
        - 총 데이터 행 수: 79,698,388 개
        - Class 0 (정상인 날): 79,641,737 개 (99.93%)
        - Class 1 (고장 임박): 56,651 개 (0.07%)
        - 실제 타겟 데이터 불균형 비율 (Class 1 : 0) = 1 : 1405.8
        
        === [4. 컬럼별 결측치 집계 (전체 열)] ===
                          Missing_Count  Missing_Ratio(%)
        smart_183_raw            643857          0.807867
        smart_184_raw            643843          0.807849
        smart_189_raw            643843          0.807849
        smart_191_raw            643843          0.807849
        smart_187_raw            643835          0.807839
        timeout_total            643835          0.807839
        smart_190_raw            643835          0.807839
        timeout_5s               643835          0.807839
        smart_241_raw            643834          0.807838
        smart_242_raw            643834          0.807838
        smart_10_raw             643823          0.807824
        Total_Seeks              643823          0.807824
        smart_198_raw            643823          0.807824
        smart_193_raw            643823          0.807824
        seek_error_count         643823          0.807824
        smart_4_raw              643823          0.807824
        smart_3_raw              643823          0.807824
        smart_199_raw            643823          0.807824
        smart_192_raw            643823          0.807824
        smart_197_raw                47          0.000059
        Total_Reads                  47          0.000059
        smart_5_raw                  47          0.000059
        smart_9_raw                  47          0.000059
        smart_194_raw                47          0.000059
        date                          0          0.000000
        serial_number                 0          0.000000
        failure                       0          0.000000
        
        === [5. 열 별 간단한 기초 통계 (Numeric Data)] ===
                 column_name column_type         min              max                         avg                   std
        0      serial_number     VARCHAR    S3000A9T         Z307Y2X9                         NaN                   NaN
        1               date        DATE  2013-05-10       2025-03-13  2018-12-03 17:57:16.998821                   NaN
        2        smart_3_raw      BIGINT           0            10590         0.03669847528729049    12.868347441464088
        3        smart_4_raw      BIGINT           1            30092          11.379990111892969    135.17048733607095
        4        smart_5_raw      BIGINT           0            65488           6.258733591455812     312.2209587739599
        5        smart_9_raw      BIGINT           0            78175          30905.879550064914     20269.12116244334
        6       smart_10_raw      BIGINT           0           262144        0.005803270184334073     32.12870716175001
        7      smart_183_raw      BIGINT           0            64730            3.28137920393203     179.4619903956368
        8      smart_184_raw      BIGINT           0              333        0.010213960500310261    0.5971151020876532
        9      smart_187_raw      BIGINT           0            65535          0.4031310884775985    61.880417991481934
        10     smart_189_raw      BIGINT           0            65535           3.128280088134085     307.0485473313004
        11     smart_191_raw      BIGINT           0          1967599           1.267518420351417    429.14639982566064
        12     smart_192_raw      BIGINT           0            30005          2.2735257223918697     134.8660418565826
        13     smart_193_raw      BIGINT           1          1581930          31981.087960385335    50088.166238432204
        14     smart_197_raw      BIGINT           0            65312          0.5984284666603035     86.81145044979996
        15     smart_198_raw      BIGINT           0            65312           0.585893211859429     84.87418327087757
        16     smart_199_raw      BIGINT           0            11534          1.4491102442977202     68.79426019448928
        17     smart_241_raw      BIGINT           0     199023000000          42063431087.659195    19529017555.087357
        18     smart_242_raw      BIGINT         234  120701120197255          209417265959.08783    378260490686.41986
        19       Total_Reads      BIGINT           0        244140624          121872413.16616563     70461470.00364728
        20  seek_error_count      BIGINT           0            65535          3.1137767945469057    415.29004513649284
        21       Total_Seeks      BIGINT           0       4294442148           465282428.4418848     289694937.4520153
        22           failure      BIGINT           0                1       0.0007108173881760319  0.026651681667939988
        23     timeout_total      BIGINT           0            63680          0.9278961579859922    113.67378058130164
        24        timeout_5s      BIGINT           0            21067          0.3928797624091303     79.20851438897874
        25     smart_190_raw      BIGINT          11               97           24.42909014993735     4.883111312823633
        26     smart_194_raw      BIGINT          11               97           24.41146281827874     4.874396579419348
        
        === [6. 시계열 연속성(Date Gap) 검사] ===
        ✅ 모든 개체의 날짜가 하루도 빠짐없이 연속적입니다.
        
        모든 종합 검증 완료. 총 소요 시간: 24.03초
        ```
        

---

### 데이터 분할 후

`(훈련:검증1:검증2:테스트 = 6:1:1:2)` 그룹 층화 분할

- train_raw.parquet (훈련)
    - https://drive.google.com/file/d/1bXc2SSWI21UXGcuvpeJdyDZCgwyoXp-h/view?usp=sharing
- val_tune_raw.parquet (검증1)
    - https://drive.google.com/file/d/1tk26QJyUT_PbjtPt2JENEsO24BgaS670/view?usp=sharing
- val_calib_raw_parquet (검증2)
    - https://drive.google.com/file/d/1wJTlQyk1Im89Qxi5l9lcxs8fPlfiONfh/view?usp=sharing
- test_raw.parquet (테스트)
    - https://drive.google.com/file/d/1KcCDTXmx6PHILRLkMzGQx9zyTx4k6DzI/view?usp=sharing

---

### 파생 변수 생성

train_raw.parquet으로부터 파생 변수를 생성한 후 그룹 층화 언더샘플링함.

- rfe_train.parquet
    - https://drive.google.com/file/d/1xJ18niwWzdvBWHkMqzGCAO9Z73q4c--e/view?usp=sharing
- rfe_test.parquet
    - https://drive.google.com/file/d/1fIe2Pu_mxQOAHHm2QX1rxncSGzYEd6WD/view?usp=sharing
</aside>

---

## **1. 데이터 정제**

연구 대상 단일 모델 추출 후 데이터 정리

- 모델 추출
    
    원본 데이터에서 `model`이 ST4000DM000인 모든 행을 추출함
    
- 이후 아래의 공정을 거침

```python
# 상수 열 (분산 0) 삭제 
# 정규화 열 삭제
# 결측치 90% 이상 열 삭제
# 완전 중복 행 삭제
# 데이터 타입을 알맞게 수정
# 객체 별 타임라인으로 정렬
```

- 중복 날짜가 있는지 확인함. (없음)
- 디코딩 및 이상치 처리
    - [smart_1_raw](https://www.notion.so/smart_1_raw-33814366e96b800cb943c3f8df0aca0e?pvs=21) , [smart_7_raw](https://www.notion.so/smart_7_raw-33814366e96b8043b155eeab57d40028?pvs=21)  디코딩
        
        <aside>
        
        Seagate 디스크의 SMART 1번과 7번 Raw 값은 '에러 횟수'와 '총 누적 작업량'이 48비트 패킹됨
        
        - **디코딩 연산 (비트 시프트)**
            
            10진수 원시값을 16진수로 변환 후, 비트 연산을 통해 상위 16비트(실제 에러 횟수, `>> 32`)와 하위 32비트(총 작업 횟수, `& 0xFFFFFFFF`)로 각각 분리 추출함.
            
        - **V3 데이터셋 반영 결과**
            
            `smart_1_raw, smart_7_raw` 원본을 삭제하고, 비트 연산을 거친 변수 3개 `Total_Reads`, `Total_Seeks`, `seek_error_count`를 생성함.
            
        </aside>
        
    
    ```python
    # smart_1 삭제 → Read_Error_Count, Total_Reads
    # smart_7 삭제 → seek_error_count, Total_Seeks
    # - Read_Error_Count는 분산이 0이라 삭제함
    ```
    
    - [smart_188_raw](https://www.notion.so/smart_188_raw-33414366e96b80538a2ada639bd6f5f4?pvs=21)  디코딩
        
        <aside>
        
        188번 원시 값은 지연 심각도별 횟수(전체, 5초 이상, 7.5초 이상)가 16비트씩 3구간으로 쪼개져 압축되어 있음. 
        
        - **디코딩 연산 (비트 3단 분리)**
            
            원시 값을 비트 연산하여 3단계로 분해 추출함: `& 0xFFFF` (하위: 전체 초과), `>> 16 & 0xFFFF` (중간: 5초 지연), `>> 32 & 0xFFFF` (상위: 7.5초 지연).
            
        - **V3 데이터셋 반영 결과**
            
            `smart_188_raw`를 삭제하고, 3가지 변수(`timeout_total`, `timeout_5s`, `Timeout_7_5s`)로 생성함.
            
        </aside>
        
    
    ```python
    # smart_188 삭제 → timeout_total, timeout_5s, Timeout_7_5s
    # Timeout_7_5s 삭제 (timeout_5s와 극도로 높은 상관성, 응답 지연이 5초를 넘긴 것부터 매우 심각한 경우)
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
    - 고장으로부터 10일 내의 구간`D-1 ~ D-10`을 모두 1(고장 임박)로 레이블링
    - 고장 당일 `D-DAY` 는 삭제 (오늘 고장나는지를 예측하는 것이 아니기 때문)

---

## 2. 결측치 보간

- HDD 개체의 일 단위 기록 누락 처리
    
    ```python
    # - **Case 1)** 마지막이 1로 끝났다가 다시 기록되는 경우    
    #     → 해당 개체 삭제
        
    # - **Case 2)** 마지막이 0으로 끝났다가 다시 기록되는 경우   
    #     → 1일 공백은 보간, 2일 이상 공백부터는 공백 전과 공백 후를 다른 시리얼 넘버를 부여. 
    				# (공백 후의 데이터들에 대해 기존 시리얼넘버 + _1씩 붙여주는 네이밍 규칙)    
    #     - 끊겨있던 구간이 포함되지 않게 파생변수를 생성하게 되는 효과 유도
    #     - 이후 데이터 분할 단계에서는 이를 같은 개체로 간주하여 분할함
    
    # - **Case 3)** 중간에 마지막이 0으로 끝난 경우   
    #     → 마지막 10일 삭제 (라벨 오염 방지 조치)
    ```
    
- 사이에 빈 시계열 행 생성
- 일괄 Forward fill 적용

<aside>

여전히 결측치가 남아있다면 파생 변수 생성 단계에서 자연스럽게 처리됨 

- Forward fill 적용 후에 결측치가 남아있다면 무조건 시작 부분임
- 결측치가 포함된 범위에서 파생변수 생성 시 NaN을 반환할 것이며, 이후 Dropna 적용하여 일괄삭제
</aside>

- 데이터 컬럼
    
    
    | **이름** | **설명** | **비고** |
    | --- | --- | --- |
    | **smart_5_raw** | 불량 섹터 대체 횟수 |  |
    | **smart_184_raw** | 데이터 전송 경로 오류 |  |
    | **smart_187_raw** | 복구 불가 오류 횟수 |  |
    | **smart_197_raw** | 불안정 섹터 수 |  |
    | **smart_198_raw** | 복구 불가 섹터 수 |  |
    | **timeout_5s** | 5초 초과 응답 지연 횟수 | 188에서 파생됨 |
    | **Timeout_7_5s** | 7.5초 초과 응답 지연 횟수 | 188에서 파생됨 |
    | **timeout_total** | 전체 응답 지연 횟수 | 188에서 파생됨 |
    | **seek_error_count** | 보이스 코일 액추에이터 탐색 오류 | 7에서 파생됨 |
    | **smart_9_raw** | 누적 사용 시간 |  |
    | **smart_189_raw** | 헤드 정렬 불량 쓰기 횟수 |  |
    | **smart_191_raw** | 외부 충격 감지 횟수 |  |
    | **smart_194_raw** | 드라이브 현재 온도 | 100도 이상은 ffill 적용 |
    | **smart_199_raw** | 케이블/통신 연결 오류 수 |  |
    | **smart_241_raw** | 누적 데이터 쓰기량 | 값 매우 큼 |
    | **smart_242_raw** | 누적 데이터 읽기량 | 값 매우 큼 |
    | **Total_Reads** | 총 읽기 섹터 시도 횟수 | 1에서 파생됨 |
    | **Total_Seeks** | 총 탐색 작업 횟수 | 7에서 파생됨 |
    | **smart_3_raw** | 모터 가동 소요 시간 |  |
    | **smart_4_raw** | 모터 온오프 횟수 |  |
    | **smart_10_raw** | 모터 재가동 시도 횟수 |  |
    | **smart_183_raw** | SATA 속도 저하 횟수 |  |
    | **smart_190_raw** | 내부 기류 온도 | 100도 이상은 ffill 적용 |
    | **smart_192_raw** | 비정상 전원 차단 횟수 |  |
    | **smart_193_raw** | 헤드 파킹 횟수 |  |
    | **smart_240_raw** | 헤드 실제 동작 시간 | 해독 방법이 명확하지 않음 |
    | **smart_1_raw** | 읽기 오류율 | [삭제] 48비트 2구간 |
    | **smart_7_raw** | 탐색 오류율 | [삭제] 48비트 2구간 |
    | **smart_12_raw** | 전원 온오프 횟수 | [삭제] 분포 확인 |
    | **smart_188_raw** | 명령 응답 지연 횟수 | [삭제] 48비트 3구간 |

---

## 3. 데이터 분할

- ~~관측 종료일을 기준으로 그룹 기반 `훈련 : 검증 : 테스트 = 6 : 2 : 2`홀드아웃 분할을 적용.~~  → 적용 불가
    
    <aside>
    
    마지막 관측일 순으로 정렬 후 10등분하여 고장 개체 수를 카운트함
    
    ```python
    10% 구간(Decile) 고장 개체 수:
    
    - D01 `2605`, D02 `443`, D03 `178`, D04 `259`, D05 `1302`
    - D06 `762`, D07 `64`, D08 `49`, D09 `19`, D10 `61`
    ```
    
    고장날 개체는 과거에 이미 죽고 살아남은 개체들은 최근에 몰려있음.
    
    생존자 편향이 극심하여 검증, 테스트 데이터셋에는 고장 개체가 수십건밖에 없음.
    
    →  평가지표가 불안정해지며 통계적 유의성을 보장받지 못하게 됨
    
    </aside>
    

그룹 층화 분할

- 개체를 구분하여 시간과 상관없이 랜덤 층화 분할 `훈련 : 검증 : 보정 : 테스트 = 6 : 1 : 1 : 2`
    - 장단점 및 의미
        
        <aside>
        
        ### 1. 잃는 것 (Trade-off)
        
        - **시간적 내러티브 (엄격한 OOT 모사 포기):** "과거의 데이터로 학습해서 미래의 고장을 예측했다"는 직관적인 달력 기반의 시간 흐름 서사는 포기해야 합니다.
        
        ### 2. 얻는 것 (Gains)
        
        - **평가의 무결성 및 통계적 유의성:** Test 셋에 충분하고 균등한 고장 개체(약 1,148개)를 강제 배정하여, 운에 따라 널뛰지 않는 '진짜 성적표'를 얻게 됩니다.
        - **치명적 편향의 완벽한 중화:** 특정 시기에 몰려 죽은 '초기 불량(Lot) 편향'과 끝까지 살아남은 '생존자 편향'을 모델 학습에서 완전히 배제합니다.
        - **개체 누수(Data Leakage) 원천 차단:** Train과 Test 셋 간의 시리얼 넘버(Entity)가 100% 격리되어 모델의 커닝을 막습니다.
        
        ### 3. 결과가 의미하는 것 (Core Implication)
        
        - **순수 물리적 고장 징후의 일반화:** 이 파이프라인에서 나온 최종 성적표는 단순히 특정 시기의 하드디스크를 잘 맞췄다는 뜻이 아닙니다. **"이 모델은 하드디스크가 언제 제조되었든, 수명이 한 달이든 10년이든 상관없이, '기계가 죽기 직전에 내뿜는 본질적인 물리적 전조증상(S.M.A.R.T. 패턴)' 그 자체를 완벽하게 학습하고 일반화했다"**는 가장 강력하고 학술적인 증명이 됩니다.
        </aside>
        
    - 4부분으로 분할하는 이유
        
        <aside>
        
        임계값 과적합 방지를 위해
        
        ### 🚨 전통적인 3분할의 치명적 문제점
        
        보통은 검증 셋(Validation) 하나를 가지고 다음 두 가지 작업을 모두 수행합니다.
        
        1. **Optuna 튜닝:** "어떤 하이퍼파라미터가 가장 성능(PR-AUC)이 좋지?"
        2. **임계값(Threshold) 설정:** "오탐률(FPR) 0.2%를 맞추려면 컷오프를 0.85로 해야겠군!"
        
        **[임계값 과적합의 위험성]**
        하이퍼파라미터를 튜닝하는 과정에서, 모델은 이미 해당 검증 셋의 '특정 노이즈와 패턴'에 알게 모르게 최적화(Overfitting)되어 버립니다. 즉, 모델에게 검증 셋은 이미 '익숙한 문제집'이 된 상태입니다.
        이 익숙해진 문제집을 바탕으로 "임계값을 0.85로 하면 오탐률이 0.2% 방어된다!"라고 확정 지은 뒤, 이를 태어나서 처음 보는 **Test 셋(실전)에 적용하면 오탐률이 1%, 2%로 미친 듯이 폭발**해 버립니다. 실무 관제 화면이 깡통 알람으로 도배되는 것이죠.
        
        </aside>
        
    - 각 분할 데이터 별 설명
        
        <aside>
        
        ```python
        "train_raw": 0.6,
        "val_tune_raw": 0.1,
        "val_calib_raw": 0.1,
        "test_raw": 0.2,
        ```
        
        1. train_raw (60%)
        
        - 고장 표본: 약 3,400개
        - 역할: 복잡한 파생변수 패턴을 깊고 확실하게 학습하기 위한 물량 확보.
        
        2. val_tune_raw (10%) : 하이퍼파라미터 튜닝용
        
        - 고장 표본: 약 570개
        - 역할: Optuna 튜닝 전용. 파라미터 탐색 시 평가지표(PR-AUC)가 요동치지 않고 안정적으로 수렴하도록 유도.
        
        3. val_calib_raw (10%) : 임계값 탐색용
        
        - 고장 표본: 약 570개
        - 역할: 튜닝에 오염되지 않은 독립된 셋. 오직 `목표 오탐률 n% 제약과 MCC 최대화의 최적 조합`을 달성하는 최적의 임계값을 정밀하게 도출하여, 실무 환경 도입 시 오탐 폭발을 원천 차단.
        
        4. test_raw (20%) : 최종 성능 평가용
        
        - 고장 표본: 약 1,100개
        - 역할: 충분한 고장 표본을 통한 모델 성능의 통계적 신뢰도 확보. 1천 개 이상의 충분한 고장 데이터로 증명하는 흔들림 없는 최종 성적표.
        </aside>
        
- 주의: 앞서 2일 이상의 공백은 시리얼번호_n 형식으로 분리했지만, 여기서는 하나의 개체로 취급해야 함
    - 물리적 개체 누수 방지 처리
        1. **물리적 동일성:** 이름표만 `A`와 `A_1`로 찢어졌을 뿐, 실제로는 공장에서 똑같이 찍혀 나온 **"완벽하게 똑같은 금속 하드디스크 1개"**입니다. 고유의 진동, 제조상 노이즈 등 물리적 특성(Signature)이 완전히 같습니다.
        2. **커닝(데이터 누수) 방지:** 만약 이 둘을 남남으로 취급하면, `A`는 수명이 짧아 Train 셋에 들어가고 나중에 부활한 `A_1`은 Test 셋에 들어가게 됩니다. 그러면 모델은 Train에서 이 하드디스크 특유의 물리적 패턴을 미리 외워버립니다(개체 누수).
        3. 결론: Test 셋은 반드시 '태어나서 한 번도 본 적 없는 낯선 디스크'만을 통과시켜야 합니다. 따라서 파생된 `_1`, `_2` 꼬리표들을 모두 원조 시리얼 넘버(Base) 기준으로 묶어 단일 그룹(Group)으로 간주하고, 무작위 층화 분할을 수행할 때 가족 단위 전체가 Train, Val, Test 중 한 곳으로 통째로 몰아져서 배정되도록(Family Binding) 처리하여 데이터 누수를 완벽히 차단합니다.
        
- 데이터 분할 결과
    
    ```python
    [설정]
    - split_ratio: {'train_raw': 0.6, 'val_tune_raw': 0.1, 'val_calib_raw': 0.1, 'test_raw': 0.2}
    - random_seed: 42
    - 엔진: DuckDB(parquet scan/집계/저장) + Pandas(최종 merge)
    - 기준: 시간 무관 랜덤 층화 + 개체(물리 디스크) 단위 그룹 분할
    - 누수 방지: serial_number의 _n 꼬리표를 base serial로 통합
    
    [개체 기준 분할 요약]
            split  total_entities  failed_entities  failed_ratio
         test_raw            7406             1149      0.155144
        train_raw           22214             3445      0.155082
    val_calib_raw            3702              574      0.155051
     val_tune_raw            3702              574      0.155051
    
    [row 기준 분할 요약(참고)]
            split  total_rows  failed_rows  failed_row_ratio
         test_raw    15954479        11293          0.000708
        train_raw    47875242        34028          0.000711
    val_calib_raw     8004501         5663          0.000707
     val_tune_raw     7864166         5667          0.000721
    
    [누수 체크]
    한 개체가 속한 split 최대 개수: 1 (정상은 1)
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_random_group_stratified\test_raw.parquet
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_random_group_stratified\train_raw.parquet
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_random_group_stratified\val_calib_raw.parquet
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_random_group_stratified\val_tune_raw.parquet
    ```
    

---

## 4. 피처 엔지니어링

자세한 내용은 [파생변수](https://www.notion.so/34b14366e96b8051a73ec7aa7cbad935?pvs=21) 참고

- 모델의 고장 예측력을 극대화하기 위해, 원본 데이터를 기반으로 다양한 관점의 파생 변수를 생성
- 파생 변수 생성은 일률적인 통계 접근법(SSP)과 도메인 지식 기반 접근법(AFM) 두 가지 트랙으로 나누어 진행

<aside>

### 파생 변수 생성 전략

- **SSP (Standard Statistical Profile): 일률적 통계 특징 추출**
    - **설명:** 원본 데이터의 시계열적 변동성을 파악하기 위해 모든 주요 원본 및 차분 변수(오늘-어제)에 일괄 적용하는 통계적 기법입니다.
    - **윈도우 적용:** 3일, 7일, 14일, 28일 주기의 슬라이딩 윈도우(Sliding Window)를 적용합니다.
    - **추출 통계량:** 변수의 성격에 따라 불필요한 연산을 제외하고 `max`, `sum`, `평균`, `표준편차`, `z-score`, `asfd`, `dai`, `cid`, `accel`, `ewma` 등을 선별적으로 계산합니다.
- **AFM (Advanced Failure Mechanism): 도메인 기반 특징 공학**
    - **설명:** 하드디스크의 물리적 동작 원리와 고장 메커니즘에 대한 도메인 지식을 바탕으로 직접 고안한 복합 파생 변수입니다.
    - **주요 지표:** 과거 물리적 손상 발생 이력(0/1 플래그), 시간당 작업량, 에러 발생 밀도 등 모델이 직접 학습하기 어려운 복합적인 패턴을 수치화합니다.

### 변수 명명 규칙 (Naming Convention)

일관된 데이터 파이프라인 관리를 위해 아래의 명명 규칙을 엄격하게 적용합니다. (전체 소문자 및 언더바 연결)

- **SSP 변수:** `[대상]_[기간]_[연산자]` 구조 (예: `s5_7d_mean`)
- **AFM 변수:** 여러 지표가 혼합되거나 특정 윈도우가 없는 경우, 변수의 직관적 물리적/논리적 의미를 담아 명명합니다. (예: `s187_14d_burst_index`)

### 데이터 결측 및 수치 예외 처리 규칙

데이터 변환 과정에서 발생할 수 있는 수치적 폭발(Gradient Explosion)과 정보 손실을 막기 위해 아래의 원칙을 적용했습니다.

**① ZeroDivisionError 방지 규칙**

분모가 0이 되는 상황을 방지하기 위해 변수의 데이터 타입에 따라 처리 방식을 이원화했습니다.

- **이산형 데이터 (빈도, 횟수, 개수 등): `+ 1` 처리 (라플라스 스무딩, Laplace Smoothing)**
    - *사유:* 미관측 범주에 대한 Zero-Probability 문제를 해결합니다. 이산형 데이터의 분모가 0일 때 아주 작은 엡실론을 더하면 결과값이 수십만 배로 폭발하여 트리 모델에 치명적인 스케일 붕괴(과적합)를 일으킬 수 있으므로 1을 더해 안정화합니다.
- **연속형 데이터 (비율, 표준편차 등): `+ ε` 처리 (수치적 안정화, Numerical Stabilization)**
    - *사유:* 미세한 소수점을 다루는 연속형 데이터에 상수 1을 더하면, 원래 데이터가 가진 예민한 변동성 신호(Signal Masking)가 훼손됩니다. 따라서 정보 손실을 최소화하기 위해 극소값 엡실론(1e-5)을 더해 연산 오류만 방지합니다.

**② 시계열 윈도우 기간 부족 (Cold Start) 처리**

- 윈도우 연산 시 과거 데이터가 부족한 구간은 초기 계산 시 `NULL`을 포함하여 계산한 후, 최종 단계에서 `0`으로 일괄 치환(Imputation)합니다.
- 초기 데이터가 불안정한 구간임을 모델이 인지할 수 있도록 별도의 Flag 변수(`is_warmup` 등)를 도입합니다.
- 차분(Difference) 데이터의 경우, 관측 첫날의 결측치는 `0`으로 처리하여 노이즈를 억제합니다.

### 피처 생성 파이프라인 및 최종 산출물

- **차분 계산 → SSP 윈도우별 계산 → AFM 계산** 순서로 진행
- 모든 산출물 파일은 데이터 압축 효율이 높은 `zstd` 포맷의 Parquet 형식으로 저장
- `serial_number`와 `date`를 복합 식별키(Composite Key)로 사용
- 최종적으로 총 268개의 파생변수가 생성됨

| **파일명** | **포함 내역 및 설명** | **변수 개수** |
| --- | --- | --- |
| `rfe_sample_diff.parquet` | **기초 데이터**
원본 데이터(19개) +  차분(18개) - failure - date/serial_number   | 40 - 3 = 37개 |
| `rfe_sample_ssp_7d.parquet` | **7일 SSP 순수 통계량**
7일 윈도우가 적용된 핵심 속성(184, 194, 241, 242, Reads, Seeks, 190)의 통계량 | 56 - 2 = 54개 |
| `rfe_sample_ssp_14d.parquet` | **14일 SSP 순수 통계량**
가장 많은 속성(17개)이 포함된 14일 주기 단기/중기 통계 데이터 | 56 - 2 = 54개 |
| `rfe_sample_ssp_28d.parquet` | **28일 SSP 순수 통계량**
하드디스크의 장기 노화 상태 및 누적 피로도를 나타내는 28일 주기 데이터 | 56 - 2 = 54개 |
| `rfe_sample_afm_windowed.parquet` | **윈도우 기반 복합체**
윈도우형 AFM(20개) | 22 - 2 = 20개 |
| `rfe_sample_afm_daily_status.parquet` | **디스크 상태 이력 (정적/이력)**
손상 여부 플래그 및 장애 발생 후 경과일 등 | 24 - 2 = 22개 |
| `rfe_sample_afm_daily_impact.parquet` | **디스크 부하 보고서 (동적/수치)**
에러 밀도, 작업량 비율 등 수치적으로 모델링된 부하 지표 | 29 - 2 = 27개 |
| **Total** | **RFE 투입 피처 총 개수 (중복 키 제외 순수 변수)** | **283 - 15 = 268개** |
</aside>

---

## **5. 변수 선택**

#### rfe_train, rfe_test 파일 생성

```markdown
# RFE용 데이터셋 제작
1. rfe_train 8 : rfe_test 2
2. serial_number 단위에서 failure 비율 유지하여 분할
3. 각 세트에 배정받은 시리얼에 한해 랜덤시드를 사용하여 정상 행 10배수 샘플링
    - train_seed = 1
    - test_seed = 2
- (같은 serial_number는 train과 test에 동시에 존재하면 안 됨)
```

- RFE 전 변수 제거
    - RFE는 만능이 아님
    - RFE는 변수 기여도를 기반으로 작동하기 때문에 지표가 희석될 수 있음
    - 너무 정보량이 적은 변수, 하위호환 변수 제거
- 특성 선택 필요성
    
    <aside>
    
    - 변수 중요도 희석 경계
        - 트리 기반 모델이 다중공선성에 강하긴 하지만, XAI 분석 시 변수 중요도가 희석될 가능성 존재
    - 연산 부담
    - 노이즈 과적합 차단
    - XAI 해석력 증가
    - 공통적으로 총 컬럼의 개수는 15~25개 사이가 좋다고 하는데 근거?
        
        ### A. 코어(Core) 결함 변수는 원래 소수입니다.
        
        - 가장 유명한 클라우드 스토리지 기업인 Backblaze의 공식 통계와 Stanford 대학의 연구(CS229 프로젝트 등)에 따르면, 수십 개의 SMART 속성 중 **실제 하드디스크의 물리적 고장과 직접적인 상관관계를 가지는 핵심 변수는 5~7개 내외**입니다. (대표적으로 SMART 5, 187, 188, 197, 198이 '운명의 5대 변수'로 불립니다).
        - 여기에 온도(190, 194), 누적 작업량(Reads/Seeks) 등 보조 지표를 합쳐도 원본 변수 기준으로는 10~15개면 고장 징후를 설명하는 데 충분합니다.
        
        ### B. 정보량의 포화 상태 (Elbow Point)
        
        - **선행 연구 트렌드:** IEEE나 MDPI 등에 게재된 SSD/HDD 고장 예측 관련 연구들을 살펴보면, Feature Selection(유전 알고리즘, 전진 선택법, RFE 등)을 수행했을 때 성능 지표(F1-Score, AUC) 그래프가 **변수 15개~20개 부근에서 고점을 찍고 평탄해지는 현상(Elbow Effect)**이 명확하게 관찰됩니다.
        - 즉, 상위 20개 내외의 변수가 모델이 고장을 예측하는 데 필요한 **전체 정보량(Information Gain)의 95% 이상을 제공**합니다. 그 이상의 변수를 투입해 보았자 성능 향상은 0.1% 미만에 불과하며, 오히려 앞서 언급한 다중공선성과 노이즈 때문에 오탐률(FPR)이 상승하는 역효과가 발생합니다.
        
        ### C. 파생 변수의 통제
        
        - 준태 님은 현재 차분 변수, 이동 평균, 누적합 등 다양한 시계열 파생 변수를 생성할 계획이십니다. 핵심 변수 10개에 대해 3가지 방식의 파생 변수만 만들어도 컬럼은 단숨에 40개로 폭발합니다.
        - RFE를 돌리면 원본 `smart_5`와 `smart_5_3일_이동평균` 중 예측력이 미세하게 더 높은 단 하나만 살아남고 나머지는 제거됩니다. 이런 식으로 불필요한 시계열 중복 특성을 솎아내고 가장 강력한 특성만 남기면 자연스럽게 15~25개 선으로 최적화됩니다.
    </aside>
    
- 재귀적 특성 제거(RFE, Recursive Feature Elimination)
    
    <aside>
    
    RFE는 이름 그대로 "전체 변수에서 시작해 가장 쓸모없는 변수를 반복적으로(Recursive) 쳐내는(Elimination)" 후진 제거(Backward Selection) 기법입니다. 이를 실제 코드와 파이프라인 관점에서 전개하면 다음과 같은 루프를 돕니다.
    
    **[RFE Pipeline Loop]**
    
    1. **초기 모델 학습:** 생성한 모든 파생 변수(예: 60개)를 포함한 훈련 셋으로 LightGBM 평가기(Estimator)를 학습시킵니다.
    2. **중요도 평가:** 학습된 모델에서 각 변수의 기여도(Feature Importance)를 추출하여 줄을 세웁니다.
    3. **최하위 변수 제거:** 중요도가 가장 낮은 변수(가장 기여하지 못한 변수)를 $N$개 제거합니다. (보통 1개씩 제거하면 시간이 너무 오래 걸리므로, 파이프라인 효율을 위해 하위 2~3개씩 Step 단위로 묶어서 제거하기도 합니다.)
    4. **성능 검증:** 남은 변수들로 다시 교차 검증(CV)을 수행하여 목적 지표(PR-AUC 또는 MCC)를 측정하고 기록합니다.
    5. **재귀적 반복:** 변수의 개수가 목표치(예: 15~25개)에 도달하거나, 성능 지표가 급격히 하락하는 임계점(Elbow Point)을 만날 때까지 1~4단계를 반복합니다.
    
    ---
    
    1. **RFE 평가기(Estimator)용 LightGBM 파라미터 전략**
    
    RFE 안에서 돌아가는 LightGBM은 **'최종 예측을 수행할 무거운 정예 모델'이 아니라, '어떤 변수가 쓸모없는지를 빠르게 판별하는 스카우터(Scouter)'** 역할을 해야 합니다.
    
    RFE 과정 자체만으로도 수십 번의 모델 학습을 반복해야 하므로, 파라미터는 철저히 **'속도(가벼움)'와 '불균형 대처'**에 초점을 맞추어 고정(Static) 값으로 설정하는 것이 정석입니다.
    
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
    
- **RFECV 기반 최적 피처 추출 프로세스핵심 요약**
    
    <aside>
    
    > **핵심 요약**
    초기 생성된 모든 후보 피처를 대상으로 모델을 반복 학습시키며, 중요도가 낮은 피처를 순차적으로 제거하여 **PR-AUC 성능을 극대화하는 최적의 피처 조합(Target: 20개)**을 찾아내는 과정
    > 
    
    ### 1️⃣ 데이터 샘플링 및 분할 전략
    
    - **언더샘플링 (Under-sampling):** 6,000만 행의 방대한 데이터에서 정보 손실을 최소화하며 연산 효율을 확보하기 위해 모든 고장 데이터를 뽑은 뒤 **정상:고장 = 10:1** 비율로 정상 데이터 샘플링 (약 26만 행).
    - **StratifiedGroupKFold (5-Fold):**
        - **Group:** 동일 디스크의 데이터가 훈련/검증셋에 섞여 데이터 누수(Data Leakage)가 발생하는 것을 방지하기 위해 `serial_number` 기준 그룹화.
        - **Stratified:** 고장(Label 1) 샘플의 극심한 불균형을 고려하여 각 폴드 내 클래스 비율 유지.
    
    ### 2️⃣ 반복적 피처 제거 (Recursive Elimination)
    
    - **Step 1 (초기 학습):** 전체 후보 피처를 입력하여 모델(LightGBM/XGBoost) 학습.
    - **Step 2 (중요도 산출):** 학습 과정에서 도출된 **변수 중요도(Feature Importance)**를 기준으로 피처 순위 부여.
    - **Step 3 (제거 및 기록):** * 중요도가 낮은 피처를 순차적으로 제거 (초반 효율을 위해 `step=10`으로 빠르게 제거 후, 후반부 `step=1`로 정밀 탐색).
        - 매 단계마다 **5-Fold 교차 검증을 통한 평균 PR-AUC** 기록.
    
    **3️⃣ 최적 지점(Elbow Point) 결정**
    
    - **성능 곡선 분석:** 피처 수($X$)에 따른 PR-AUC($Y$) 그래프 생성
    - **Elbow Method 적용:** 피처 수가 줄어듦에도 PR-AUC 성능이 하락하지 않거나, 급격한 하락이 시작되기 직전의 변곡점(Elbow)을 최적의 피처 개수로 선정.
    - **윈도우 최적화 자동화:** 동일 센서의 여러 윈도우(7, 14, 21, 28일) 중 목적 함수(10일 내 고장 예측)에 가장 적합한 기간의 변수가 자동 생존.
    
    ### 4️⃣ 최종 피처 확정 (Target: ~20개)
    
    - 모델의 **해석력(Explainability)**과 **운영 효율성**을 고려하여 성능 손실이 미미한 범위 내에서 최소한의 피처셋(약 20개) 확정.
    - 최종 선택된 피처들을 대상으로 전체 훈련 데이터셋 재학습 및 모델 고도화.
    </aside>
    

### 1. 변수 grouping

```markdown
## A. 디스크 물리적 손상 / Bad Sector 계열

### 직접 손상 플래그

- s5_damaged
- s187_damaged
- s197_damaged
- s198_damaged
- seek_damaged
- timeout_5s_damaged

### 과거 발생 여부

- s5_ever_flag
- s187_ever_flag

### 최초/최근 발생 시점

- s5_days_since_first
- s187_days_since_first
- s191_days_since_last
- s199_days_since_last
- timeout_total_days_since_last

→ “damage persistence / recency”

---

## B. Read / Seek / Timeout / IO 불안정성

### 기본 I/O 이상

- age_weighted_seek_error
- io_asymmetry_index
- seek_error_density
- timeout_read_density
- timeout_seek_density
- timeout_severity_ratio
- timeout_to_uncorrectable_lag1

### Spike / burst

- seek_error_14d_spike_ratio
- seek_spike_ratio
- read_spike_ratio
- write_spike_ratio
- uncorrectable_spike_ratio

### write 안정성

- write_stability_ratio

→ “I/O path degradation”

---

## C. Reallocation / Pending / Uncorrectable

### sector deterioration

- reallocated_pending_ratio
- pending_to_offline_ratio
- cumulative_error_score
- error_growth_ratio
- error_saturation_score
- multi_error_coincidence

→ “latent sector collapse”

---

## D. Shock / Vibration / Mechanical Stress

### shock 기반

- shock_fatigue_rate
- shock_seek_interaction
- log_shock_fly_interaction
- shock_to_highfly_ratio

### seek + shock

- fatal_crash_interaction

→ “mechanical degradation”

---

## E. Thermal / Temperature Stress

- thermal_stress_index
- temp_error_index
- thermal_fatigue_integral_7d

→ “thermal degradation”

---

## F. Firmware / Recovery Failure / Cascading Failure

- firmware_struggle_index
- cascading_failure_flag
- recovery_failure_flag
- data_corruption_hazard
- s197_recovery_flag
- s184_1d_crash_flag

→ “systemic failure risk”

---

## G. Workload / Usage Intensity

- age_weighted_workload
- workload_intensity
- workload_7d_accel

→ usage pressure

---

## H. 특정 SMART raw counter 파생

대표적으로:

- s184_*
- s190_*
- s194_*
- s241_*
- s242_*
- total_reads_*
- total_seeks_*

그리고 각각:

- max
- sum
- mean
- std
- asfd
- cid
- dai
- zscore
- ewma

→ “time-series statistical descriptors”

이건 별도 군집으로 봐야 합니다.
```

### 2. near-zero varience 제거

### 3. group 내부 중복 제거

- Spearman
- 도메인 지식

### 4. failure / non-failure 분포 차이 검정

- Mann-Whitney U
- KS test

### 5. RFE

- 변수 기여도(gain)를 기준으로 하위권 변수들을 재귀적으로 탈락시킴
- 매 진행마다 PR-AUC를 기록하여 그래프를 그림.
1. 전체 변수에 대해 100개가 남을 때까지 10개씩 탈락시킴
2. 50개가 남을 때까지 5개씩 탈락시킴
3. 이후 1개씩 탈락시키며 그래프의 엘보우 포인트를 확인하여 결정함
    - 만약 확실한 지점이 없다면 목표 변수 개수(20개) 전후로 결정함

---

## **5. 모델 훈련**

- 모델 훈련
    
    <aside>
    
    극단적인 클래스 불균형(약 1 : 1400)을 해소하고 메모리 한계를 극복하기 위해, 모델 훈련 시 정상 데이터에 한하여 언더샘플링을 적용
    
    - 모든 정상 데이터를 학습에 사용하지 않는 논리적 근거
        - 시계열적 중복성 (Temporal Redundancy): 고장이 임박한 디스크는 데이터 분포의 변화가 뚜렷하지만, 건강한 디스크는 수백 일 동안 데이터의 변화가 거의 없습니다. 매일 연속된 정상 데이터를 모두 학습시키는 것은 동일한 노이즈를 중복 학습시키는 것에 불과합니다.
        - 선행 연구 기반 타당성: 관련 연구에 따르면, 정상 데이터의 시계열 중복성으로 인해 무작위 다운샘플링을 통해 전체의 10%만 유지하더라도 모든 샘플을 사용할 때와 비교하여 예측 정확도가 하락하지 않는 것으로 증명되었습니다. 이를 통해 네트워크 대역폭과 연산 오버헤드를 획기적으로 줄일 수 있습니다.
        - 앙상블 연산 비용의 물리적 한계: 소수 클래스인 고장 임박 데이터는 약 5만 6천 개입니다. 전체 8천만 개의 정상 데이터를 한 번씩이라도 모두 소진하려면 최소 1,400개 이상의 훈련 세트($n$)를 생성하여 앙상블을 수행해야 합니다. 여기에 Optuna 하이퍼파라미터 튜닝까지 결합될 경우, 실무적으로 감당할 수 없는 막대한 컴퓨팅 비용과 시간이 발생합니다.
    - 샘플링 방식: 행(Row) 단위 무작위 추출
        - 정상 데이터를 덜어낼 때는 개체(Entity) 단위가 아닌, 훈련 셋 전체의 '행(Row) 단위' 무작위 추출을 수행
        - 특정 정상 디스크의 지루한 1년 치 데이터를 통째로 넣는 대신, 수만 대의 다양한 하드디스크가 가진 다채로운 정상 패턴을 골고루 수집함으로써 앙상블 모델의 다양성을 극대화
        - (중복 데이터를 줄이고 다양성을 확보)
    </aside>
    
- 앙상블 최적화
    
    <aside>
    
    비대칭 언더배깅 및 하이퍼파라미터 최적화
    
    - 학습 모델: LightGBM
        - 빠른 연산 속도와 우수한 성능을 자랑하는 트리 기반의 LightGBM을 채택
    - 훈련 방법론: 비복원 추출 기반의 비대칭 언더배깅(Asymmetric Underbagging)
        - 복원 추출 시 발생할 수 있는 과적합을 방지하기 위해
    - 최적의 언더샘플링 규모 경험적 탐색
        - 훈련셋의 언더샘플링 비율에 따른 검증 셋의 PR-AUC 점수 궤적을 기록하고 추적
        - 정상 데이터의 투입량을 늘린다고 해서 성능이 무한정 비례하여 상승하지 않으며, 특정 시점부터는 연산 비용만 급증하는 정보량 포화 지점이 존재함
        - 따라서 PR-AUC가 최고점에 도달하거나 성능 향상폭이 급격히 둔화되는 엘보우 포인트(Elbow Point)를 식별하여, '연산 비용 최소화'와 '예측 성능 극대화'의 최적 균형점을 갖춘 샘플링 규모를 과학적으로 선정
    - Optuna 기반 튜닝 (PR-AUC 최적화)
        - 선정된 최적의 데이터 규모를 바탕으로, 앙상블 세트의 개수(n), 클래스 가중치, 그리고 LightGBM의 내부 하이퍼파라미터를 Optuna로 동시 탐색
        - 목적 함수로는 임계값(Threshold)의 변동에 영향을 받지 않는 절대적 성능 지표인 PR-AUC를 최대화하도록 설정하여 모델의 본질적인 분류 능력을 끌어올림
    </aside>
    
- 앙상블 통합 및 실무형 임계값(Threshold) 최적화
    
    <aside>
    
    - 결과 결합: n개의 개별 LightGBM 모델이 산출한 예측 확률들을 소프트 보팅(Soft Voting) 방식으로 결합하여 최종 고장 확률을 산출
    - 오탐률 제약 기반 임계값 탐색: 앙상블의 예측 확률에 대해 그리드 서치(Grid Search)를 수행하여, 비즈니스 요구사항인 오탐률 제약조건(FPR 0.1% ~ 0.5% 등 타겟 수치 확정 필요)을 엄격히 만족하는 선에서 MCC(매튜스 상관계수)가 최대화되는 최적 임계값을 찾음 (이 과정은 시각화 그래프와 함께 도출)
        - 만약 그리드 서치 탐색 결과 어떤 임계값에서도 목표 오탐률 제약조건을 달성하지 못할 경우, 알고리즘(Optuna) 목적 함수에 오탐률 페널티를 부여하고 모델을 전면 재학습
    </aside>
    

---

## 6. 모델 평가

- 최종 테스트 시에는 실무 환경과 동일하게 매일 1일씩 전진하는 슬라이딩 윈도우를 적용해 일일 롤링 추론 수행.
- 기술적 평가(성적표)는 행(Row)별로 다 소진해서 계산하되, 실무적 평가(보고서)는 개체(Entity)별 첫 탐지 시점을 기준으로 요약
    1. **모델 평가 리포트 (행 단위)**
        
        테스트 셋 전체 행에 대해 임계값을 적용하여 혼동 행렬(Confusion Matrix)을 구합니다.
        
        - **지표:** MCC, PR-AUC, F1-Score
        - **비고:** 모델의 수학적 성능 증명용.
    2. **실무 운영 리포트 (개체 단위)**
        
        각 시리얼 넘버별로 시간순 정렬 후, **'최초 알람 발생 시점'**을 기준으로 분석합니다.
        
        - **고장 탐지율 (Hit Rate):** 전체 고장 개체 중 $D-10 \sim D-1$ 사이에 알람이 한 번이라도 울린 개체의 비율.
        - **평균 사전 경보 시간 (Avg. Lead Time):** 알람이 울린 개체들이 평균적으로 고장 몇 일 전에 알려줬는가?
        - **오탐 개체 수:** 고장이 아닌데 알람이 울린 '시리얼 넘버'의 개수. (실무자들은 하루에 몇 개의 '깡통 알람'이 발생하는지에 민감합니다.)
- 불균형 데이터의 특성상 정확도보다는 F1-score, 오탐률, PR-AUC, 매튜스 상관계수(MCC)에 중점.
- 모델 평가 체계: OOT 분할과 일일 롤링 추론(Rolling Inference)의 실무적 무결성
    - **데이터 무결성 (OOT Split)**: 랜덤 분할이 아닌 절대적 시간축(6:2:2)을 기준으로 데이터를 나누어, 미래를 미리 보고 과거를 맞추는 미래 참조 누수를 차단함.
    - **실무 재현성 (Rolling Inference)**: 매일 자정 로그를 수집하고 추론하는 실제 관제 환경과 동일하게 **1일 단위 슬라이딩 윈도우** 기법을 적용하여, 모델의 일관된 방어력을 검증함.
    - **평가 엄격성 (MCC & PR-AUC)**: 정확도(Accuracy)나 ROC-AUC의 착시를 버리고, 1:1400의 극단적 불균형 데이터에서 진성 고장 탐지와 오탐 최소화를 엄중히 따지는 지표에 집중함.
- 완화된 평가 윈도우(Relaxed Evaluation Window)
    
     기법의 도입이 필수적이다. 모델 파라미터의 학습 및 가중치 업데이트 시에는 명확한 열화 패턴이 짙은 D-10 구간을 타겟 변수로 강제하더라도, 추론 및 평가 프레임워크 상에서는 D-30부터 D-1 사이의 어느 선행 시점에서든 알람을 발생시켜 유지보수 시간을 벌어주었다면 이를 모두 정탐(True Positive)으로 승격하여 간주하는 평가 룰(Rule)을 서술
    

---

## 7. 모델 해석

- SHAP: 어떤 요소 때문에 고장이 났는가를 규명
- **Global(전역) 및 Local(국소) 이중 해석 체계**
    - **전역적 해석 (Global Interpretability):** SHAP Summary Plot 등을 이용해 전체 하드디스크 집단에서 어떤 SMART 변수(예: SMART 5, 187 등)가 예측 전반에 가장 큰 영향을 미치는지 거시적인 트렌드를 분석합니다.
    - **국소적 해석 (Local Interpretability):** 현장 엔지니어가 "특정 시리얼 넘버(예: S3008532_1)의 디스크 알람이 오늘 왜 울렸는가?"를 직관적으로 납득할 수 있도록, SHAP Waterfall Plot이나 LIME을 사용해 개별 디스크의 예측 근거(예: "온도는 정상이지만 명령 시간 초과 횟수가 급증하여 고장 확률 85%로 판판됨")를 개별적으로 설명하는 파이프라인을 추가합니다.
- 시계열 의존성(Temporal Dependence) 분석
    
    하드디스크 고장은 하루아침에 일어나는 것이 아니라 시간 흐름에 따른 점진적인 열화(Degradation) 과정입니다. 따라서 단순히 정적인 SHAP 값이 아니라, '시간의 흐름에 따른 SMART 지표의 변화 궤적'이 모델의 예측 확률에 어떻게 기여했는지를 보여주는 시계열 중심의 SHAP 의존성 분석을 수행하겠다고 보강하면 시계열 데이터라는 특성을 완벽하게 살릴 수 있습니다.
    
    → 잘 맞춘 고장 샘플 1~2개로 마지막 30일 그래프 그리기 (국소적 해석)
    
- 실무 적용 시 기대효과


# 파생변수 

최근 부하 압력	is_warmup_28d	$\mathbb{I}(t_{elapsed} < 28)$	28일 관측 구간 부족 플래그	초기 관측 구간(노이즈) 필터링
최근 부하 압력	is_warmup_14d	$\mathbb{I}(t_{elapsed} < 14)$	14일 관측 구간 부족 플래그	초기 관측 구간(노이즈) 필터링
최근 부하 압력	is_warmup_7d	$\mathbb{I}(t_{elapsed} < 7)$	7일 관측 구간 부족 플래그	초기 관측 구간(노이즈) 필터링
최근 부하 압력	workload_intensity	$\frac{s9_t + 1}{\Delta s241_t + \Delta s242_t + 1}$	누적 사용 시간 대비 총 작업량 비율	나이(누적치) 대비 당일 총 작업량 비율
최근 부하 압력	age_weighted_workload	$\ln(\lvert\Delta s241_t + \Delta s242_t\rvert + 1) \times \ln(s9_t + 1)$	노후도 기반 일일 작업 부하 가중치	작업 감소(음수) 시 에러 방지용 절댓값 적용
최근 부하 압력	workload_7d_accel	$(\Delta s241_t + \Delta s242_t) - (\Delta s241_{t-7} + \Delta s242_{t-7})$	총 작업량 7일 가속도	Temporal Diff (유지)
총 탐색량	total_seeks_28d_ewma	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	총 탐색 작업 28일 지수가중	28일 장기적 기계 탐색 트렌드를 완만히 따라가는 이동평균 지표
총 탐색량	total_seeks_28d_dai	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	총 탐색 작업 28일 평균증가량	28일 만성적 추세에서 보여지는 일평균 탐색 증감 기울기
총 탐색량	total_seeks_28d_cid	$\sqrt{\sum_{i=0}^{27}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	총 탐색 작업 28일 불변거리	28일 장기 시계열 상의 구동 복잡성과 요동 거리 스케일
총 탐색량	total_seeks_28d_asfd	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 탐색 작업 28일 차분절대합	28일간 잦은 탐색 횟수 변동으로 쌓인 만성적 물리 구동 피로합
총 탐색량	total_seeks_28d_zscore	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	총 탐색 작업 28일 Z스코어	28일 만성 기저 평균을 넘어선 당일의 헤드 탐색 비정상 수치
총 탐색량	total_seeks_28d_std	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	총 탐색 작업 28일 차분 편차	28일 장기 만성적인 기계 헤드 동작 횟수의 산포 불규칙성
총 탐색량	total_seeks_28d_mean	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	총 탐색 작업 28일 차분 평균	28일 장기간에 걸쳐 평균화된 만성 기계 탐색 기저 부하
총 탐색량	total_seeks_28d_sum	$\sum_{i=0}^{27} \Delta W_{t-i}$	총 탐색 작업 28일 차분 총합	28일 만성 장기 구동 과정에서 누적된 기계 탐색 부하 총량
총 탐색량	total_seeks_28d_max	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	총 탐색 작업 28일 차분 최댓값	28일 내 헤드 구동 기계 부하가 집중된 장기 최대 스파이크 일
총 탐색량	total_seeks_14d_ewma	$0.133 \cdot \Delta W_t + 0.867 \cdot E_{t-1}$	총 탐색 작업 14일 지수가중	14일 중기 탐색 작업 패턴을 부드럽게 지시하는 트렌드 지표
총 탐색량	total_seeks_14d_dai	$\frac{\Delta W_t - \Delta W_{t-13}}{13}$	총 탐색 작업 14일 평균증가량	14일 중기 탐색 횟수의 전반적 증감 추세 기울기
총 탐색량	total_seeks_14d_cid	$\sqrt{\sum_{i=0}^{13}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	총 탐색 작업 14일 불변거리	14일 중기 탐색 부하 시계열의 구조적 복잡도
총 탐색량	total_seeks_14d_asfd	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 탐색 작업 14일 차분절대합	14일 중기 탐색 횟수의 지속적인 널뛰기로 인한 변동 피로도
총 탐색량	total_seeks_14d_zscore	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	총 탐색 작업 14일 Z스코어	14일 중기 기저 상태를 반영한 당일 탐색 횟수 이상 스케일
총 탐색량	total_seeks_14d_std	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	총 탐색 작업 14일 차분 편차	14일 중기 탐색 동작의 요동치는 분산 수준
총 탐색량	total_seeks_14d_mean	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	총 탐색 작업 14일 차분 평균	14일간 중기 일평균 헤드 탐색 기계 동작 횟수
총 탐색량	total_seeks_14d_sum	$\sum_{i=0}^{13} \Delta W_{t-i}$	총 탐색 작업 14일 차분 총합	14일 중기적으로 누적된 헤드 탐색 이동 부하 데미지
총 탐색량	total_seeks_14d_max	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	총 탐색 작업 14일 차분 최댓값	14일 내 헤드 구동 액추에이터의 발생 최대 트래픽 스파이크
총 탐색량	total_seeks_7d_ewma	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	총 탐색 작업 7일 지수가중	7일 내 탐색 트래픽 추세에 지수적으로 민감하게 반응하는 평균
총 탐색량	total_seeks_7d_dai	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	총 탐색 작업 7일 평균증가량	7일간 매일 기계 탐색 작업이 늘거나 줄어든 선형 추세
총 탐색량	total_seeks_7d_cid	$\sqrt{\sum_{i=0}^{6}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	총 탐색 작업 7일 불변거리	7일 단기 헤드 구동 기계 부하 시계열의 복잡성 거리
총 탐색량	total_seeks_7d_asfd	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 탐색 작업 7일 차분절대합	7일간 헤드 탐색 횟수의 급증/급감 누적 플래핑(Flapping)
총 탐색량	total_seeks_7d_zscore	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	총 탐색 작업 7일 Z스코어	과거 7일 탐색 부하 대비 당일 탐색 시도의 스파이크 지수
총 탐색량	total_seeks_7d_std	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	총 탐색 작업 7일 차분 편차	7일간 헤드 탐색 작업의 불규칙한 산포도
총 탐색량	total_seeks_7d_mean	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	총 탐색 작업 7일 차분 평균	7일간 단기 일평균 헤드 탐색 구동 부하 빈도
총 탐색량	total_seeks_7d_sum	$\sum_{i=0}^{6} \Delta W_{t-i}$	총 탐색 작업 7일 차분 총합	7일간 액추에이터 기계 부품이 수행한 탐색 작업 총량
총 탐색량	total_seeks_7d_max	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	총 탐색 작업 7일 차분 최댓값	7일 내 디스크 헤드가 트랙을 가장 많이 찾아다닌 일일 피크
총 읽기량	total_reads_28d_ewma	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	총 읽기 시도 28일 지수가중	28일 장기적 동작 패턴 트렌드를 부드럽게 추종하는 추세
총 읽기량	total_reads_28d_dai	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	총 읽기 시도 28일 평균증가량	28일 장기간에 걸친 평균적 동작 빈도 증감 추세 기울기
총 읽기량	total_reads_28d_cid	$\sqrt{\sum_{i=0}^{27}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	총 읽기 시도 28일 불변거리	28일간 장기 동작 부하 시계열의 복잡한 요동 거리
총 읽기량	total_reads_28d_asfd	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 읽기 시도 28일 차분절대합	28일간 동작 빈도 널뛰기로 인해 축적된 장기 피로합
총 읽기량	total_reads_28d_zscore	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	총 읽기 시도 28일 Z스코어	28일 장기 평균 기저 상태 대비 당일 읽기 동작 이례성
총 읽기량	total_reads_28d_std	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	총 읽기 시도 28일 차분 편차	28일 장기 읽기 동작 횟수의 불규칙 분산 폭
총 읽기량	total_reads_28d_mean	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	총 읽기 시도 28일 차분 평균	28일 장기간 평균적으로 감당해온 만성 동작 부하
총 읽기량	total_reads_28d_sum	$\sum_{i=0}^{27} \Delta W_{t-i}$	총 읽기 시도 28일 차분 총합	28일 동안 드라이브 구동부에 가해진 장기 누적 동작 수
총 읽기량	total_reads_28d_max	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	총 읽기 시도 28일 차분 최댓값	28일 내 가장 집중적으로 기계가 동작한 날의 횟수 스파이크
총 읽기량	total_reads_14d_ewma	$0.133 \cdot \Delta W_t + 0.867 \cdot E_{t-1}$	총 읽기 시도 14일 지수가중	14일간 읽기 횟수 증감 트렌드의 지수가중 이동평균
총 읽기량	total_reads_14d_dai	$\frac{\Delta W_t - \Delta W_{t-13}}{13}$	총 읽기 시도 14일 평균증가량	14일 중기 일일 동작 횟수 선형 추이 기울기
총 읽기량	total_reads_14d_cid	$\sqrt{\sum_{i=0}^{13}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	총 읽기 시도 14일 불변거리	14일 중기 읽기 동작 부하 시계열의 요동치는 수준
총 읽기량	total_reads_14d_asfd	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 읽기 시도 14일 차분절대합	14일 중기 읽기 동작 횟수의 급변 피로도 거리
총 읽기량	total_reads_14d_zscore	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	총 읽기 시도 14일 Z스코어	14일 중기 동작 기준치 대비 당일 작업 폭주 스케일
총 읽기량	total_reads_14d_std	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	총 읽기 시도 14일 차분 편차	14일간 읽기 동작 빈도의 중기적 불규칙 산포도
총 읽기량	total_reads_14d_mean	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	총 읽기 시도 14일 차분 평균	14일간 디스크 구동부의 중기 일평균 동작 부하 빈도
총 읽기량	total_reads_14d_sum	$\sum_{i=0}^{13} \Delta W_{t-i}$	총 읽기 시도 14일 차분 총합	14일간 가해진 중기 누적 순수 읽기 횟수 총합
총 읽기량	total_reads_14d_max	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	총 읽기 시도 14일 차분 최댓값	14일 내 발생한 하루 기준 읽기 동작 횟수의 최대 피크
총 읽기량	total_reads_7d_ewma	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	총 읽기 시도 7일 지수가중	7일 내 읽기 동작 빈도 증가 트렌드의 단기 지수가중
총 읽기량	total_reads_7d_dai	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	총 읽기 시도 7일 평균증가량	7일간 단기 일일 동작 횟수 증감 추세선 기울기
총 읽기량	total_reads_7d_cid	$\sqrt{\sum_{i=0}^{6}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	총 읽기 시도 7일 불변거리	7일간 기계 동작 부하의 시계열 복잡도
총 읽기량	total_reads_7d_asfd	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 읽기 시도 7일 차분절대합	7일간 동작 횟수 증가/감소가 누적된 총 굴곡 거리
총 읽기량	total_reads_7d_zscore	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	총 읽기 시도 7일 Z스코어	과거 7일 동작 평균 대비 당일 횟수 폭주 이상치
총 읽기량	total_reads_7d_std	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	총 읽기 시도 7일 차분 편차	7일간 읽기 동작 횟수의 기계적 불규칙 산포
총 읽기량	total_reads_7d_mean	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	총 읽기 시도 7일 차분 평균	7일간 일평균 디스크 읽기 동작 횟수
총 읽기량	total_reads_7d_sum	$\sum_{i=0}^{6} \Delta W_{t-i}$	총 읽기 시도 7일 차분 총합	7일간 드라이브에 가해진 순수 읽기 동작 실행 총 횟수
총 읽기량	total_reads_7d_max	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	총 읽기 시도 7일 차분 최댓값	7일 내 하루 기준 디스크 암이 읽기 동작을 수행한 횟수 피크
총 읽기량	s242_28d_ewma	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	누적 읽기량 28일 지수가중	28일 장기 읽기 부하의 부드러운 지수가중 트렌드
총 읽기량	s242_28d_dai	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	누적 읽기량 28일 평균증가량	28일 장기간 읽기 트래픽의 선형 추세 기울기
총 읽기량	s242_28d_cid	$\sqrt{\sum_{i=0}^{27}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	누적 읽기량 28일 불변거리	28일간 읽기 I/O가 얼마나 요동치며 요청되었는가
총 읽기량	s242_28d_asfd	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 읽기량 28일 차분절대합	28일 읽기 트래픽의 변화 총량 및 장기 굴곡 피로도
총 읽기량	s242_28d_zscore	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	누적 읽기량 28일 Z스코어	28일 장기 기저 상태 대비 당일 읽기 폭주 감지
총 읽기량	s242_28d_std	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	누적 읽기량 28일 차분 편차	28일간 만성적인 장기 읽기 부하의 불안정성
총 읽기량	s242_28d_mean	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	누적 읽기량 28일 차분 평균	28일간 디스크가 지속 감당한 기저 읽기 부하 상태
총 읽기량	s242_28d_sum	$\sum_{i=0}^{27} \Delta W_{t-i}$	누적 읽기량 28일 차분 총합	28일 동안 기록된 만성 장기 총 읽기 데이터량
총 읽기량	s242_28d_max	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	누적 읽기량 28일 차분 최댓값	28일 내 발생한 하루 최대 읽기 데이터 스파이크
총 읽기량	s242_14d_ewma	$0.133 \cdot \Delta W_t + 0.867 \cdot E_{t-1}$	누적 읽기량 14일 지수가중	최근 14일 내 읽기량 트렌드 민감 지수가중 평균
총 읽기량	s242_14d_dai	$\frac{\Delta W_t - \Delta W_{t-13}}{13}$	누적 읽기량 14일 평균증가량	14일간 중기 일평균 읽기 부하 선형 트렌드
총 읽기량	s242_14d_cid	$\sqrt{\sum_{i=0}^{13}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	누적 읽기량 14일 불변거리	14일 중기 읽기 트래픽의 시계열 복잡도 측정
총 읽기량	s242_14d_asfd	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 읽기량 14일 차분절대합	14일간 잦은 읽기 요청 증감의 피로도 누적
총 읽기량	s242_14d_zscore	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	누적 읽기량 14일 Z스코어	과거 14일 기준 대비 당일 읽기 부하 이상치 척도
총 읽기량	s242_14d_std	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	누적 읽기량 14일 차분 편차	14일간 읽기 부하의 중기적 불규칙 산포도
총 읽기량	s242_14d_mean	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	누적 읽기량 14일 차분 평균	14일간 디스크가 처리한 중기 일평균 읽기 부하량
총 읽기량	s242_14d_sum	$\sum_{i=0}^{13} \Delta W_{t-i}$	누적 읽기량 14일 차분 총합	14일 동안 기록된 순수 중기 총 읽기 데이터량
총 읽기량	s242_14d_max	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	누적 읽기량 14일 차분 최댓값	14일 내 발생한 하루 최대 데이터 읽기 부하 피크
총 읽기량	s242_7d_ewma	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	누적 읽기량 7일 지수가중	최근 7일 내 단기 읽기 부하 폭증 감지 추세
총 읽기량	s242_7d_dai	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	누적 읽기량 7일 평균증가량	7일간 일평균 단기 읽기 부하 상승 선형 추세
총 읽기량	s242_7d_cid	$\sqrt{\sum_{i=0}^{6}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	누적 읽기량 7일 불변거리	7일간 읽기 트래픽 널뛰기 현상 및 복잡도
총 읽기량	s242_7d_asfd	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 읽기량 7일 차분절대합	7일간 읽기 요청 급증/급감을 반복한 총 이동 거리
총 읽기량	s242_7d_zscore	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	누적 읽기량 7일 Z스코어	과거 7일 읽기량 대비 당일 폭주 수준 스케일링
총 읽기량	s242_7d_std	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	누적 읽기량 7일 차분 편차	7일간 읽기 부하의 불규칙한 산포도
총 읽기량	s242_7d_mean	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	누적 읽기량 7일 차분 평균	7일간 디스크가 처리한 단기 일평균 읽기 부하량
총 읽기량	s242_7d_sum	$\sum_{i=0}^{6} \Delta W_{t-i}$	누적 읽기량 7일 차분 총합	7일 동안 드라이브가 순수하게 읽어낸 총 데이터량
총 읽기량	s242_7d_max	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	누적 읽기량 7일 차분 최댓값	7일 내 발생한 하루 최대 데이터 읽기 부하 피크
총 기록량	s241_28d_ewma	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	누적 쓰기량 28일 지수가중	장기 28일 쓰기량 트렌드 파악 지수가중 이동평균
총 기록량	s241_28d_dai	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	누적 쓰기량 28일 평균증가량	28일간 만성적인 일일 쓰기 트래픽 증가/감소 추세
총 기록량	s241_28d_cid	$\sqrt{\sum_{i=0}^{27}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	누적 쓰기량 28일 불변거리	28일 트래픽 장기 시계열의 꼬불꼬불한 굴곡 수준
총 기록량	s241_28d_asfd	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 쓰기량 28일 차분절대합	28일간 잦은 I/O 전환으로 누적된 장기 부하 피로도
총 기록량	s241_28d_zscore	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	누적 쓰기량 28일 Z스코어	장기 28일 기준치 대비 당일 부하의 이상 수치
총 기록량	s241_28d_std	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	누적 쓰기량 28일 차분 편차	28일간 쓰기 부하의 장기 만성적 불규칙 수준
총 기록량	s241_28d_mean	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	누적 쓰기량 28일 차분 평균	28일간 디스크가 장기적으로 감당한 기저 쓰기 부하
총 기록량	s241_28d_sum	$\sum_{i=0}^{27} \Delta W_{t-i}$	누적 쓰기량 28일 차분 총합	28일 동안 기록된 만성적인 순수 쓰기 총 데이터량
총 기록량	s241_28d_max	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	누적 쓰기량 28일 차분 최댓값	28일 내 발생한 하루 최대 쓰기 데이터 스파이크
총 기록량	s241_14d_ewma	$0.133 \cdot \Delta W_t + 0.867 \cdot E_{t-1}$	누적 쓰기량 14일 지수가중	최근 14일 중기 쓰기 트렌드 지수가중 이동평균
총 기록량	s241_14d_dai	$\frac{\Delta W_t - \Delta W_{t-13}}{13}$	누적 쓰기량 14일 평균증가량	14일간 중기 일평균 쓰기 부하 상승 선형 추세
총 기록량	s241_14d_cid	$\sqrt{\sum_{i=0}^{13}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	누적 쓰기량 14일 불변거리	14일간 중기적 트래픽 복잡도 및 변동성
총 기록량	s241_14d_asfd	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 쓰기량 14일 차분절대합	14일간 중기적 쓰기 부하의 전환 피로도 거리
총 기록량	s241_14d_zscore	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	누적 쓰기량 14일 Z스코어	과거 14일 평균 대비 당일 쓰기량 폭주 통계적 척도
총 기록량	s241_14d_std	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	누적 쓰기량 14일 차분 편차	14일간 쓰기 부하량의 중기적 불규칙 산포도
총 기록량	s241_14d_mean	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	누적 쓰기량 14일 차분 평균	14일간 디스크가 처리한 중기 일평균 쓰기 부하량
총 기록량	s241_14d_sum	$\sum_{i=0}^{13} \Delta W_{t-i}$	누적 쓰기량 14일 차분 총합	14일 동안 기록된 순수 쓰기 데이터 중기 총량
총 기록량	s241_14d_max	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	누적 쓰기량 14일 차분 최댓값	14일 내 발생한 하루 최대 데이터 쓰기 부하 피크
총 기록량	s241_7d_ewma	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	누적 쓰기량 7일 지수가중	최근 7일 쓰기량 폭증에 가중치를 둔 트렌드
총 기록량	s241_7d_dai	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	누적 쓰기량 7일 평균증가량	7일간 일평균 쓰기 부하 상승 선형 추세
총 기록량	s241_7d_cid	$\sqrt{\sum_{i=0}^{6}(\Delta W_{t-i} - \Delta W_{t-i-1})^2}$	누적 쓰기량 7일 불변거리	7일간 쓰기 트래픽 시계열 복잡도 및 널뛰기 현상
총 기록량	s241_7d_asfd	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 쓰기량 7일 차분절대합	7일간 쓰기 부하의 급증/급감을 반복한 총 이동 거리
총 기록량	s241_7d_zscore	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	누적 쓰기량 7일 Z스코어	과거 7일 평균 대비 당일 쓰기량 폭주의 이상 수치
총 기록량	s241_7d_std	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	누적 쓰기량 7일 차분 편차	7일간 쓰기 부하량의 불규칙한 요동(산포도)
총 기록량	s241_7d_mean	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	누적 쓰기량 7일 차분 평균	7일간 디스크가 처리한 일평균 쓰기 부하량
총 기록량	s241_7d_sum	$\sum_{i=0}^{6} \Delta W_{t-i}$	누적 쓰기량 7일 차분 총합	7일 동안 기록된 순수 쓰기 데이터 총량
총 기록량	s241_7d_max	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	누적 쓰기량 7일 차분 최댓값	7일 내 발생한 하루 최대 데이터 쓰기 부하 피크
Burst 이상	s192_14d_burst	$\sum_{i=0}^{13} \Delta s192_{t-i}$	14일 강제 종료 단기 폭주량	[이름 수정] Density $\rightarrow$ Burst로 의미 일치
Burst 이상	s189_28d_highfly_burst	$\sum_{i=0}^{27} \Delta s189_{t-i}$	28일간 불량 쓰기 증가량	단순 변화량 윈도우 합산 (유지)
Burst 이상	s187_14d_burst_index	$\sum_{i=0}^{13} \left( \Delta s187_{t-i} \cdot \mathbb{I}(\Delta s187_{t-i} > 0) \right)$	복구 불가 오류 횟수 중기 누적 증가량	증가된(양수) 부분만 필터링 합산 (유지)
Burst 이상,Reallocated / Pending	s5_relative_score_14d	$\frac{s5_t}{P_{95}(s5_{t-14:t-1}) + 1}$	14일 최고점 대비 당일 불량 섹터 비율	[이상치 방어] $\max$가 유발하는 정규화 왜곡을 막기 위해 $P_{95}$ 백분위수로 대체
Burst 이상,급성 Spike	read_spike_ratio	$\frac{\Delta s242_t}{\left( \frac{1}{7}\sum_{i=1}^{7} \Delta s242_{t-i} \right) + 1}$	7일 평균 대비 당일 읽기량 폭주 비율	당일($t$) 제외 과거 7일 기준 (유지)
Burst 이상	s189_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	헤드 정렬 불량 쓰기 28일 총합	28일 내 발생한 기계적 구동 불안정성 누적 스트레스
Burst 이상	s189_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	헤드 정렬 불량 쓰기 28일 최댓값	28일 내 발생한 헤드 정렬 불량 쓰기의 최대 피크
Burst 이상,Sector 열화	s187_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	복구 불가 오류 14일 총합	14일 내 발생한 치명적 복구 불가 에러 누적량
Burst 이상,Sector 열화	s187_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	복구 불가 오류 14일 최댓값	14일 내 발생한 복구 불가(Uncorrectable) 에러 피크
급성 Spike	write_spike_ratio	$\frac{\Delta s241_t}{\left( \frac{1}{7}\sum_{i=1}^{7} \Delta s241_{t-i} \right) + 1}$	7일 평균 대비 당일 쓰기량 폭주 비율	$\Delta$ 기반 폭주 연산 (유지)
급성 Spike	uncorrectable_spike_ratio	$\frac{\Delta s198_t}{\left( \frac{1}{14}\sum_{i=1}^{14} \Delta s198_{t-i} \right) + 1}$	복구 불가 섹터 중기 폭증 비율	$\Delta$ 기반 폭주 연산 (유지)
급성 Spike	seek_spike_ratio	$\frac{\Delta Total\_Seeks_t}{\left( \frac{1}{7}\sum_{i=1}^{7} \Delta Total\_Seeks_{t-i} \right) + 1}$	7일 평균 대비 당일 탐색 폭주 비율	$\Delta$ 기반 폭주 연산 (유지)
급성 Spike	seek_error_14d_spike_ratio	$\frac{\max(0, \Delta Seek\Error_t)}{AVG{14d}(\max(0, \Delta Seek\Error_{t-i})) + 1}$	탐색 오류 14일 대비 이상치 비율	[노이즈 방어] 희소 이벤트 폭주를 막기 위해 분모를 SMA에서 EMA로 교체
읽기/쓰기 안정성	write_stability_ratio	$\frac{s189_t + 1}{\Delta s241_t + 1}$	쓰기량 대비 헤드 정렬 불량률	당일 쓰기량 대비 누적 헤드 정렬 불량 비율
읽기/쓰기 안정성	s183_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	SATA 속도 저하 28일 총합	28일 내 SATA 속도 저하 장기 누적 발생 횟수
읽기/쓰기 안정성	s183_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	SATA 속도 저하 28일 최댓값	28일 내 SATA 속도 저하 현상의 최대 피크
읽기/쓰기 안정성	s183_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	SATA 속도 저하 14일 총합	14일 내 SATA 속도 저하 현상 누적 발생 횟수
읽기/쓰기 안정성	s183_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	SATA 속도 저하 14일 최댓값	14일 내 SATA 속도 저하 현상의 최대 피크
Seek 경로 이상	age_weighted_seek_error	$\Delta Seek\_Error_t \times s9_t$	노후도 대비 탐색 오류 심각도	당일 발생량과 누적 사용 시간 곱셈
Seek 경로 이상	timeout_seek_density	$\frac{Timeout\_Total_t}{Total\_Seeks_t + 1}$	탐색 작업 대비 에러율 (밀도)	역방향 뒤집어 Density로 통일
Seek 경로 이상	seek_error_density	$\frac{Seek\_Error\_Count_t}{Total\_Seeks_t + 1}$	탐색당 탐색 오류 밀도	역방향 뒤집어 Density로 통일
Seek 경로 이상	seek_error_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	탐색 오류 28일 총합	28일 내 기계 구동부 탐색 실패 장기 누적 횟수
Seek 경로 이상	seek_error_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	탐색 오류 28일 최댓값	28일 내 액추에이터 탐색 실패 횟수의 최대 피크
Seek 경로 이상	seek_error_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	탐색 오류 14일 총합	14일 내 기계 구동부 탐색 실패 누적 횟수
Seek 경로 이상	seek_error_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	탐색 오류 14일 최댓값	14일 내 액추에이터 탐색 실패 횟수의 최대 피크
기본 I/O 이상	s199_days_since_last	$\begin{cases} t - \max\{\tau \mid \max(0, \Delta s199_\tau) > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	통신 오류 증가 이후 무사고 경과일	[표준 cases] 순수 증가 이벤트 기준
기본 I/O 이상	timeout_to_uncorrectable_lag1	$\max(0, \Delta Timeout\_Total_t) \times \max(0, \Delta s198_{t-1})$	시차 결합 지수	어제 불량 발생 $\times$ 오늘 지연 발생
기본 I/O 이상	timeout_severity_ratio	$\frac{Timeout\_Total_t + 1}{Timeout\_5s_t + 1}$	타임아웃 심각도 비율	전체 타임아웃 대비 치명적 지연(5s) 비율
기본 I/O 이상	timeout_read_density	$\frac{Timeout\_Total_t}{Total\_Reads_t + 1}$	읽기 작업 대비 에러율 (밀도)	역방향 뒤집어 Density로 통일
기본 I/O 이상	s199_error_density	$\frac{s199_t}{Total\_Reads_t + Total\_Seeks_t + 1}$	작업당 통신 연결 오류 밀도	역방향 뒤집어 Density로 통일
기본 I/O 이상	io_asymmetry_index	$\frac{\lvert\Delta s241_t - \Delta s242_t\rvert}{\Delta s241_t + \Delta s242_t + 1}$	읽기/쓰기 작업 비대칭 지수	작업량 차이의 절댓값을 전체로 나눔
기본 I/O 이상	s199_14d_burst	$\sum_{i=0}^{13} \Delta s199_{t-i}$	14일간 통신 오류 단기 폭주량	단순 변화량 윈도우 합산 (유지)
기본 I/O 이상	timeout_total_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	전체 지연 28일 총합	28일 내 펌웨어 응답 지연 현상 장기 누적 횟수
기본 I/O 이상	timeout_total_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	전체 지연 28일 최댓값	28일 내 발생한 타임아웃 횟수의 최대 피크
기본 I/O 이상	timeout_total_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	전체 지연 14일 총합	14일 내 펌웨어 응답 지연 현상 누적 횟수
기본 I/O 이상	timeout_total_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	전체 지연 14일 최댓값	14일 내 발생한 타임아웃 횟수의 최대 피크
기본 I/O 이상	timeout_5s_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	5초 지연 28일 총합	28일 내 치명적 펌웨어 멈춤 장기 누적 발생 횟수
기본 I/O 이상	timeout_5s_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	5초 지연 28일 최댓값	28일 내 발생한 5초 초과 I/O 펜딩의 최대 피크
기본 I/O 이상	timeout_5s_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	5초 지연 14일 총합	14일 내 치명적 펌웨어 멈춤 현상의 누적 발생 횟수
기본 I/O 이상	timeout_5s_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	5초 지연 14일 최댓값	14일 내 발생한 5초 초과 I/O 펜딩(멈춤)의 단기 피크
기본 I/O 이상	s199_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	통신 오류 28일 총합	28일 내 통신 에러 누적 발생 장기 횟수 합산
기본 I/O 이상	s199_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	통신 오류 28일 최댓값	28일 내 컨트롤러 간 통신 에러의 최대 피크
기본 I/O 이상	s199_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	통신 오류 14일 총합	14일 내 통신 에러 누적 발생 횟수 합산
기본 I/O 이상	s199_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	통신 오류 14일 최댓값	14일 내 컨트롤러 간 통신 에러(CRC)의 최대 피크
물리 스트레스 상호작용	log_shock_fly_interaction	$\ln(1 + \lvert\Delta s191_t \times \Delta s189_t\rvert) \times \text{sgn}(\Delta s191_t \times \Delta s189_t)$	외부 충격 및 헤드 비행 오류 동시 발생 심각도	로그 내 음수 충돌 방지 및 부호 복원
물리 스트레스 상호작용	fatal_crash_interaction	$\Delta Timeout\_5s_t \times \Delta s187_t$	응답 지연 및 복구 불가 오류 결합 지수	변화량 기준 단순 곱셈
물리 스트레스 상호작용	shock_to_highfly_ratio	$\frac{\sum_{i=0}^{27} \lvert\Delta s191_{t-i}\rvert + 1}{\sum_{i=0}^{27} \lvert\Delta s189_{t-i}\rvert + 1}$	충격 대비 헤드 불안정 전이 비율	[음수 상쇄 방어] 절댓값(\lvert, \rvert) 씌워서 순수 누적 충격량만 비교
기계적 충격	s191_days_since_last	$\begin{cases} t - \max\{\tau \mid \max(0, \Delta s191_\tau) > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	외부 충격 발생 이후 무사고 경과일	[표준 cases] 순수 증가 이벤트 기준
기계적 충격	shock_seek_interaction	$\Delta s191_t \times \Delta Seek\_Error_t$	충격 및 탐색 오류 동시 발생 결합 지수	당일 변화량 기준 단순 곱셈
기계적 충격	shock_fatigue_rate	$\frac{s191_t}{s9_t + 1}$	시간당 외부 충격 누적 피로도	역방향 뒤집어 Rate로 통일
기계적 충격	s191_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	외부 충격 28일 총합	28일 내 장기 누적된 외부 물리 충격 횟수 합산
기계적 충격	s191_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	외부 충격 28일 최댓값	28일 내 발생한 외부 물리적 충격의 최대치
기계적 충격	s191_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	외부 충격 14일 총합	14일 내 누적된 외부 물리 충격 횟수 합산
기계적 충격	s191_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	외부 충격 14일 최댓값	14일 내 발생한 외부 물리적 충격(G-Sense)의 최대치
열 스트레스	temp_error_index	$\max(0, s194_t - 40) \times E_t$	고온 환경 에러 발생 지수	40도 이하 강제 0 처리 후 당일 에러 총합 곱셈
열 스트레스	thermal_fatigue_integral_7d	$\sum_{i=0}^{6} \max(0, s194_{t-i} - 40)$	7일간 40도 초과 열 피로 누적량(면적)	Physics-style 적분 연산 (유지)
열 스트레스	s190_28d_ewma	$0.069 \cdot T_t + 0.931 \cdot E_{t-1}$	기류 온도 28일 지수가중	장기 28일 내 온도 변화 추이의 지수가중 이동평균
열 스트레스	s190_28d_dai	$\frac{T_t - T_{t-27}}{27}$	기류 온도 28일 평균증가량	28일간 만성적인 쿨링 성능 저하 여부 추세
열 스트레스	s190_28d_cid	$\sqrt{\sum_{i=0}^{27}(T_{t-i} - T_{t-i-1})^2}$	기류 온도 28일 불변거리	28일 온도 시계열 장기 복잡도 및 누적 열충격
열 스트레스	s190_28d_asfd	$\sum_{i=0}^{27} \lvert T_{t-i} - T_{t-i-1} \rvert$	기류 온도 28일 차분절대합	28일 장기적 온도 변화로 누적된 열 피로도 총합
열 스트레스	s190_28d_zscore	$\frac{T_t - \mu_{28}}{\sigma_{28} + 1e-5}$	기류 온도 28일 Z스코어	장기 28일 평균 대비 당일 온도의 통계적 이상 수치
열 스트레스	s190_28d_std	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(T_{t-i} - \mu_{28})^2}$	기류 온도 28일 표준편차	28일간 기류 온도의 장기 불규칙 산포도
열 스트레스	s190_28d_mean	$\frac{1}{28}\sum_{i=0}^{27} T_{t-i}$	기류 온도 28일 평균	28일간 계절성 포함 평균 내부 기류 온도 상태
열 스트레스	s190_28d_max	$\max(T_{t-27}, \dots, T_t)$	기류 온도 28일 최댓값	28일간 기록된 내부 기류 온도의 최대치
열 스트레스	s190_14d_ewma	$0.133 \cdot T_t + 0.867 \cdot E_{t-1}$	기류 온도 14일 지수가중	최근 14일 내 기류 온도 변화의 지수가중 이동평균
열 스트레스	s190_14d_dai	$\frac{T_t - T_{t-13}}{13}$	기류 온도 14일 평균증가량	14일간 일평균 내부 기류 온도 상승/하락 선형 추세
열 스트레스	s190_14d_cid	$\sqrt{\sum_{i=0}^{13}(T_{t-i} - T_{t-i-1})^2}$	기류 온도 14일 불변거리	14일간 온도 시계열의 복잡도 및 열충격 스케일
열 스트레스	s190_14d_asfd	$\sum_{i=0}^{13} \lvert T_{t-i} - T_{t-i-1} \rvert$	기류 온도 14일 차분절대합	14일간 냉각/발열을 오가며 발생한 중기 열 변화 거리
열 스트레스	s190_14d_zscore	$\frac{T_t - \mu_{14}}{\sigma_{14} + 1e-5}$	기류 온도 14일 Z스코어	과거 14일 평균 대비 당일 온도의 통계적 이상 수치
열 스트레스	s190_14d_std	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(T_{t-i} - \mu_{14})^2}$	기류 온도 14일 표준편차	14일간 내부 기류 온도의 불규칙한 산포도
열 스트레스	s190_14d_mean	$\frac{1}{14}\sum_{i=0}^{13} T_{t-i}$	기류 온도 14일 평균	14일간 평균 내부 기류 온도 상태
열 스트레스	s190_14d_max	$\max(T_{t-13}, \dots, T_t)$	기류 온도 14일 최댓값	14일간 기록된 내부 기류 온도의 최대치
열 스트레스	s190_7d_ewma	$0.25 \cdot T_t + 0.75 \cdot EWMA_{t-1}$	기류 온도 7일 지수가중	최근 7일 내 급성 냉각 문제에 가중치를 둔 추세
열 스트레스	s190_7d_dai	$\frac{T_t - T_{t-6}}{6}$	기류 온도 7일 평균증가량	7일간 일평균 내부 기류 온도 상승/하락 선형 추세
열 스트레스	s190_7d_cid	$\sqrt{\sum_{i=0}^{6}(T_{t-i} - T_{t-i-1})^2}$	기류 온도 7일 불변거리	7일간 온도 시계열의 복잡도 및 열충격 스케일
열 스트레스	s190_7d_asfd	$\sum_{i=0}^{6} \lvert T_{t-i} - T_{t-i-1} \rvert$	기류 온도 7일 차분절대합	7일간 냉각/발열을 오가며 발생한 총 열 변화 거리
열 스트레스	s190_7d_zscore	$\frac{T_t - \mu_7}{\sigma_7 + 1e-5}$	기류 온도 7일 Z스코어	과거 7일 평균 대비 당일 온도의 통계적 이상 수치
열 스트레스	s190_7d_std	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(T_{t-i} - \mu_{7})^2}$	기류 온도 7일 표준편차	7일간 내부 기류 온도의 불규칙한 산포도
열 스트레스	s190_7d_mean	$\frac{1}{7}\sum_{i=0}^{6} T_{t-i}$	기류 온도 7일 평균	7일간 평균 내부 기류 온도 상태
온도 수준	thermal_stress_index	$\max(0, s194_t - 40)^2 \times \Delta s241_t$	고온 스트레스 및 누적 쓰기량 결합 지수	40도 초과분에 가중치(제곱) 부여
온도 수준	s194_over40_7d_count	$\sum_{i=0}^{6} \mathbb{I}(s194_{t-i} > 40)$	7일 중 온도가 40도를 초과한 일수	조건 기반 카운트 (유지)
온도 수준	s194_28d_ewma	$0.069 \cdot T_t + 0.931 \cdot E_{t-1}$	드라이브 온도 28일 지수가중	장기 28일 내 발열 트렌드의 지수가중 이동평균
온도 수준	s194_28d_dai	$\frac{T_t - T_{t-27}}{27}$	드라이브 온도 28일 증가량	28일간 일평균 장기 만성 온도 상승 선형 추세
온도 수준	s194_28d_cid	$\sqrt{\sum_{i=0}^{27}(T_{t-i} - T_{t-i-1})^2}$	드라이브 온도 28일 거리	28일간 드라이브 표면에 누적된 만성적 열 충격 스케일
온도 수준	s194_28d_asfd	$\sum_{i=0}^{27} \lvert T_{t-i} - T_{t-i-1} \rvert$	드라이브 온도 28일 차분합	28일간 드라이브에 누적된 만성 열 피로도 총합
온도 수준	s194_28d_zscore	$\frac{T_t - \mu_{28}}{\sigma_{28} + 1e-5}$	드라이브 온도 28일 Z스코어	과거 28일 평균 발열 대비 당일의 장기 과열 이상치
온도 수준	s194_28d_std	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(T_{t-i} - \mu_{28})^2}$	드라이브 온도 28일 편차	28일간 온도 산포에 따른 장기간의 부하 불규칙성
온도 수준	s194_28d_mean	$\frac{1}{28}\sum_{i=0}^{27} T_{t-i}$	드라이브 온도 28일 평균	28일간 만성적인 드라이브 평균 발열 상태
온도 수준	s194_28d_max	$\max(T_{t-27}, \dots, T_t)$	드라이브 온도 28일 최댓값	28일간 기록된 드라이브 표면 온도 장기 최대치
온도 수준	s194_14d_ewma	$0.133 \cdot T_t + 0.867 \cdot E_{t-1}$	드라이브 온도 14일 지수가중	14일 내 드라이브 온도 변화의 지수가중 이동평균
온도 수준	s194_14d_dai	$\frac{T_t - T_{t-13}}{13}$	드라이브 온도 14일 증가량	14일간 일평균 드라이브 온도 선형 증가 추세
온도 수준	s194_14d_cid	$\sqrt{\sum_{i=0}^{13}(T_{t-i} - T_{t-i-1})^2}$	드라이브 온도 14일 거리	14일간 드라이브 표면에 가해진 중기 열 충격
온도 수준	s194_14d_asfd	$\sum_{i=0}^{13} \lvert T_{t-i} - T_{t-i-1} \rvert$	드라이브 온도 14일 차분합	14일간 드라이브에 발생한 중기 열 변화량 누적
온도 수준	s194_14d_zscore	$\frac{T_t - \mu_{14}}{\sigma_{14} + 1e-5}$	드라이브 온도 14일 Z스코어	과거 14일 발열 대비 당일의 중기 과열 이상치
온도 수준	s194_14d_std	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(T_{t-i} - \mu_{14})^2}$	드라이브 온도 14일 편차	14일간 드라이브 온도의 중기 산포도
온도 수준	s194_14d_mean	$\frac{1}{14}\sum_{i=0}^{13} T_{t-i}$	드라이브 온도 14일 평균	14일간 드라이브 평균 발열 상태
온도 수준	s194_14d_max	$\max(T_{t-13}, \dots, T_t)$	드라이브 온도 14일 최댓값	14일간 기록된 드라이브 표면 온도의 중기 최대치
온도 수준	s194_7d_ewma	$0.25 \cdot T_t + 0.75 \cdot EWMA_{t-1}$	드라이브 온도 7일 지수가중	7일 내 드라이브 과열 상태에 가중치를 둔 추세
온도 수준	s194_7d_dai	$\frac{T_t - T_{t-6}}{6}$	드라이브 온도 7일 증가량	7일간 일평균 드라이브 온도 선형 증가 추세
온도 수준	s194_7d_cid	$\sqrt{\sum_{i=0}^{6}(T_{t-i} - T_{t-i-1})^2}$	드라이브 온도 7일 거리	7일간 드라이브 표면에 가해진 단기 열 충격
온도 수준	s194_7d_asfd	$\sum_{i=0}^{6} \lvert T_{t-i} - T_{t-i-1} \rvert$	드라이브 온도 7일 차분합	7일간 드라이브 수축/팽창 변화의 이동 거리
온도 수준	s194_7d_zscore	$\frac{T_t - \mu_7}{\sigma_7 + 1e-5}$	드라이브 온도 7일 Z스코어	과거 7일 발열 대비 당일의 비정상적 과열 감지
온도 수준	s194_7d_std	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(T_{t-i} - \mu_{7})^2}$	드라이브 온도 7일 편차	7일간 드라이브 온도의 불안정한 요동 수준
온도 수준	s194_7d_mean	$\frac{1}{7}\sum_{i=0}^{6} T_{t-i}$	드라이브 온도 7일 평균	7일간 드라이브 평균 발열 상태
온도 수준	s194_7d_max	$\max(T_{t-6}, \dots, T_t)$	드라이브 온도 7일 최댓값	7일간 기록된 드라이브 표면 온도의 단기 최대치
시스템성 실패	cascading_failure_flag	$\mathbb{I}(\Delta s197_t > 0 \land \Delta Timeout\_Total_t > 0)$	결함 도미노 연쇄 발생 플래그	대기 섹터와 지연 시간 동시 증가
시스템성 실패	zero_to_hero_count	$\sum_{k \in \mathcal{K}} \mathbb{I}(s_{k,t} > 0 \land s_{k,t-1} = 0)$	0이었다가 갑자기 튄 지표의 개수	[집합 K 확정] 타겟 변수 명확화
시스템성 실패	s184_1d_crash_flag	$\mathbb{I}(\Delta s184_t > 0)$	치명적 패리티 오류 1일 발생 플래그	단순 발생 여부 확인
시스템성 실패	data_corruption_hazard	$\mathbb{I}(\Delta s184_t > 0 \lor \Delta s199_t > 0)$	논리적 데이터 오염 위험 경고 플래그	패리티 오류 또는 통신 오류 발생
시스템성 실패	s184_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	전송 경로 오류 14일 총합	14일 내 발생한 전송 경로 오류 장기 누적 데미지
시스템성 실패	s184_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	전송 경로 오류 14일 최댓값	14일 내 발생한 전송 경로 오류 최대 피크
시스템성 실패	s184_7d_sum	$\sum_{i=0}^{6} \Delta E_{t-i}$	전송 경로 오류 7일 총합	7일 내 발생한 전송 경로 오류 누적 데미지
시스템성 실패	s184_7d_max	$\max(\Delta E_{t-6}, \dots, \Delta E_t)$	전송 경로 오류 7일 최댓값	7일 내 발생한 전송 경로 오류 최대 피크
시스템성 실패	s184_3d_sum	$\sum_{i=0}^{2} \Delta E_{t-i}$	전송 경로 오류 3일 총합	3일 내 발생한 전송 경로 오류 급성 누적 데미지
시스템성 실패	s184_3d_max	$\max(\Delta E_{t-2}, \dots, \Delta E_t)$	전송 경로 오류 3일 최댓값	3일 내 발생한 패리티 오류(End-to-End)의 급성 피크
펌웨어 실패	firmware_struggle_index	$(\Delta s197_t + \Delta s198_t) \times \Delta Timeout\_Total_t$	펌웨어 오류 복구 지연(발악) 지수	변화량 기준 단순 곱셈
Sector 열화	s197_recovery_flag	$\mathbb{I}(s197_t < s197_{t-1})$	불안정 섹터 회복 이벤트 플래그	전일 대비 누적 수치 감소 여부
Sector 열화	s198_error_rate	$\frac{s198_t}{s9_t + 1}$	시간당 복구 불가 섹터 발생률	Inverse 뒤집어 Rate로 통일
Sector 열화	s187_error_rate	$\frac{s187_t}{s9_t + 1}$	시간당 복구 불가 오류 발생률	Inverse 뒤집어 Rate로 통일
Sector 열화	multi_error_coincidence	$\sum_{k} \mathbb{I}(\Delta s_{k,t} > 0)$	다중 오류 동시 발생 지표	값이 증가한 핵심 에러 지표 개수 카운트
Sector 열화	error_saturation_score	$\mathbb{I}(s187_t \ge 65535) + \mathbb{I}(s189_t \ge 65535) + \mathbb{I}(Seek\_Error_t \ge 65535)$	핵심 오류 지표 포화도 점수	임계치 도달 여부 확인
Sector 열화	cumulative_error_score	$5\lvert\Delta s5_t\rvert + 4\lvert\Delta s187_t\rvert + 3\lvert\Delta Timeout\_Total_t\rvert + 2\lvert\Delta s197_t\rvert + 1\lvert\Delta s198_t\rvert$	핵심 5대 변수 종합 위험 점수	에러 복구(음수)도 이벤트로 간주하여 절댓값 가중 합산
Sector 열화	error_density_14d	$\frac{\sum_{i=0}^{13} E_{t-i}}{\sum_{i=0}^{13} \max(0, \Delta s9_{t-i}) + 1}$	14일간 시간 대비 에러 밀도	[논리 위험 수정] 분모를 14일간의 실제 누적 구동 시간($\sum \Delta s9$)으로 변경
Sector 열화	s198_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	복구 불가 섹터 28일 총합	28일 내 물리적으로 파괴된 섹터 장기 누적량
Sector 열화	s198_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	복구 불가 섹터 28일 최댓값	28일 내 발생한 오프라인 복구 불가 섹터의 최대 피크
Sector 열화	s198_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	복구 불가 섹터 14일 총합	14일 내 물리적으로 파괴된 섹터의 총 누적량
Sector 열화	s198_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	복구 불가 섹터 14일 최댓값	14일 내 발생한 오프라인 복구 불가 섹터의 최대 피크
Sector 열화	s197_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	불안정 섹터 28일 총합	28일 내 대기 섹터로 전환된 장기 누적 횟수
Sector 열화	s197_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	불안정 섹터 28일 최댓값	28일 내 대기 섹터 증가량의 최대 피크
Sector 열화	s197_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	불안정 섹터 14일 총합	14일 내 대기 섹터로 전환된 총 누적 횟수
Sector 열화	s197_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	불안정 섹터 14일 최댓값	14일 내 대기 섹터(Pending) 증가량의 최대 피크
Sector 열화	s187_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	복구 불가 오류 28일 총합	28일 내 발생한 치명적 복구 불가 장기 누적량
Sector 열화	s187_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	복구 불가 오류 28일 최댓값	28일 내 발생한 복구 불가 에러 최대 피크
Reallocated / Pending	reallocated_pending_ratio	$\frac{s197_t + 1}{s5_t + 1}$	불량 섹터 대비 불안정 섹터 비율	누적 원본값 기준 비율 연산
Reallocated / Pending	pending_to_offline_ratio	$\frac{s197_t + 1}{s198_t + 1}$	대기 섹터의 복구 불가 전이 심각도	누적 원본값 기준 비율 연산
지속 악화	s5_daily_failure_speed	$\max(0, \Delta s5_t)$	일일 불량 섹터 노후화 속도	역방향 수식 버리고 당일 순수 발생량으로 치환
지속 악화	late_stage_degradation	$\max(0, \Delta s5_t) \times s9_t$	노후화 말기 불량 섹터 발생 위험도	불량 섹터 감소분은 0 처리 후 누적 시간 곱셈
지속 악화	error_growth_ratio	$\frac{E_t + 1}{E_{t-1} + 1}$	오류 가속화 비율	전일 대비 당일 에러 발생 비율
지속 악화	s197_7d_straight_rise	$\mathbb{I}\left( \sum_{i=0}^{6} \mathbb{I}(\Delta s197_{t-i} > 0) \ge 5 \right)$	7일 중 5일 이상 연속 상승 여부	이중 지시 함수를 통한 논리 연산 (유지)
복구 실패	recovery_failure_flag	$\mathbb{I}(\Delta s197_t < 0) \cdot \mathbb{I}(\Delta s5_t > 0)$	대기 섹터의 불가 섹터 전이 현상	[트리 최적화] Binary Interaction 유지
최근 발생 시점	timeout_total_days_since_last	$\begin{cases} t - \max\{\tau \mid \max(0, \Delta Timeout\_Total_\tau) > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	마지막 Timeout 증가 이후 경과일	[표준 cases] 순수 증가 이벤트 기준
최초 발생 시점	s5_days_since_first	$\begin{cases} t - \min\{\tau \mid s5_\tau > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	smart_5 최초 발생 후 경과일	[표준 cases] 미발생 시 -1 반환
최초 발생 시점	s187_days_since_first	$\begin{cases} t - \min\{\tau \mid s187_\tau > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	smart_187 최초 발생 후 경과일	[표준 cases] 미발생 시 -1 반환
과거 누적 발생	s5_ever_flag	$\mathbb{I}(\max_{0 \le i \le t}(s5_i) > 0)$	불량 섹터 누적 발생 플래그	과거부터 현재까지 발생 이력 확인
과거 누적 발생	s187_ever_flag	$\mathbb{I}(\max_{0 \le i \le t}(s187_i) > 0)$	복구 불가 오류 횟수 누적 발생 플래그	과거부터 현재까지 발생 이력 확인
직접 손상 발생	timeout_5s_damaged	$\mathbb{I}(Timeout\_5s_t > 0)$	5초 이상 지연 손상 여부	누적 원본값 기준 발생 확인
직접 손상 발생	seek_damaged	$\mathbb{I}(Seek\_Error\_Count_t > 0)$	탐색 오류 손상 여부	누적 원본값 기준 발생 확인
직접 손상 발생	s5_damaged	$\mathbb{I}(s5_t > 0)$	불량 섹터 손상 여부	누적 원본값 기준 발생 확인
직접 손상 발생	s198_damaged	$\mathbb{I}(s198_t > 0)$	복구 불가 섹터 손상 여부	누적 원본값 기준 발생 확인
직접 손상 발생	s197_damaged	$\mathbb{I}(s197_t > 0)$	불안정 섹터 손상 여부	누적 원본값 기준 발생 확인
직접 손상 발생	s187_damaged	$\mathbb{I}(s187_t > 0)$	복구 불가 오류 손상 여부	누적 원본값 기준 발생 확인
직접 손상 발생	s5_28d_sum	$\sum_{i=0}^{27} \Delta E_{t-i}$	불량 섹터 28일 총합	28일 내 발생한 불량 섹터 대체 횟수 장기 누적 데미지
직접 손상 발생	s5_28d_max	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	불량 섹터 28일 최댓값	28일 내 발생한 불량 섹터 대체 횟수의 최대 피크
직접 손상 발생	s5_14d_sum	$\sum_{i=0}^{13} \Delta E_{t-i}$	불량 섹터 14일 총합	14일 내 발생한 불량 섹터 대체 횟수 누적 데미지
직접 손상 발생	s5_14d_max	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	불량 섹터 14일 최댓값	14일 내 발생한 불량 섹터 대체 횟수의 최대 피크