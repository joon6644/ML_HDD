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
          serial_number       date  smart_3_raw  smart_4_raw  smart_5_raw  smart_9_raw  smart_10_raw  smart_183_raw  smart_184_raw  smart_187_raw  smart_189_raw  smart_191_raw  smart_192_raw  smart_193_raw  smart_197_raw  smart_198_raw  smart_199_raw  smart_241_raw  smart_242_raw  Total_Reads  seek_error_count  total_seeks  failure  timeout_total  Timeout_5s  smart_190_raw  smart_194_raw
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
        Timeout_Total            643835          0.807839
        smart_190_raw            643835          0.807839
        Timeout_5s               643835          0.807839
        smart_241_raw            643834          0.807838
        smart_242_raw            643834          0.807838
        smart_10_raw             643823          0.807824
        total_seeks              643823          0.807824
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
        21       total_seeks      BIGINT           0       4294442148           465282428.4418848     289694937.4520153
        22           failure      BIGINT           0                1       0.0007108173881760319  0.026651681667939988
        23     Timeout_Total      BIGINT           0            63680          0.9278961579859922    113.67378058130164
        24        Timeout_5s      BIGINT           0            21067          0.3928797624091303     79.20851438897874
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
- val_calib_raw.parquet (검증2)
    - https://drive.google.com/file/d/1wJTlQyk1Im89Qxi5l9lcxs8fPlfiONfh/view?usp=sharing
- test_raw.parquet (테스트)
    - https://drive.google.com/file/d/1KcCDTXmx6PHILRLkMzGQx9zyTx4k6DzI/view?usp=sharing

---

### 파생 변수 생성

train_raw.parquet으로부터 파생 변수를 생성한 후 8:2 그룹 층화 언더샘플링함 (클래스 불균형은 다름).

- fs_sample_train.parquet
    - [https://drive.google.com/file/d/11fU3EnwZPIIWyDp-AYiPkxP8g59VIHOm/view?usp=sharing](https://drive.google.com/file/d/1-AReRkLyZIko11HDexNLfqakuXuQaaJ1/view?usp=sharing)
- fs_sample_test.parquet
    - [https://drive.google.com/file/d/1d5rcAzEOiiDwaq3v0w9O-2D2NQji58e3/view?usp=sharing](https://drive.google.com/file/d/1LRo5pJnb5svoN8fhZyhawon2kh-05FaE/view?usp=sharing)

---

### 1차 필터링 후 RFE 전용 데이터셋

group 내부 고상관 특성을 필터링한 결과물

- fs_train.parquet
    - [https://drive.google.com/file/d/1LjgwwRVbfmJ8gCxUieBly0QurWUCTT40/view?usp=sharing](https://drive.google.com/file/d/1ryADNuOrGtR4bzV8Ha7mdm5xsyr7oegp/view?usp=sharing)
- fs_test.parquet
    - https://drive.google.com/file/d/13qsAKr9m1S9mH1rYwDchVnvtOItCCGAH/view?usp=sharing
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
    - [smart_1_raw](https://www.notion.so/smart_1_raw-33814366e96b800cb943c3f8df0aca0e?pvs=21) , ‣  디코딩
        
        <aside>
        
        Seagate 디스크의 SMART 1번과 7번 Raw 값은 '에러 횟수'와 '총 누적 작업량'이 48비트 패킹됨
        
        - **디코딩 연산 (비트 시프트)**
            
            10진수 원시값을 16진수로 변환 후, 비트 연산을 통해 상위 16비트(실제 에러 횟수, `>> 32`)와 하위 32비트(총 작업 횟수, `& 0xFFFFFFFF`)로 각각 분리 추출함.
            
        - **V3 데이터셋 반영 결과**
            
            `smart_1_raw, smart_7_raw` 원본을 삭제하고, 비트 연산을 거친 변수 3개 `Total_Reads`, `total_seeks`, `seek_error_count`를 생성함.
            
        </aside>
        
    
    ```python
    # smart_1 삭제 → Read_Error_Count, Total_Reads
    # smart_7 삭제 → seek_error_count, total_seeks
    # - Read_Error_Count는 분산이 0이라 삭제함
    ```
    
    - [smart_188_raw](https://www.notion.so/smart_188_raw-33414366e96b80538a2ada639bd6f5f4?pvs=21)  디코딩
        
        <aside>
        
        188번 원시 값은 지연 심각도별 횟수(전체, 5초 이상, 7.5초 이상)가 16비트씩 3구간으로 쪼개져 압축되어 있음. 
        
        - **디코딩 연산 (비트 3단 분리)**
            
            원시 값을 비트 연산하여 3단계로 분해 추출함: `& 0xFFFF` (하위: 전체 초과), `>> 16 & 0xFFFF` (중간: 5초 지연), `>> 32 & 0xFFFF` (상위: 7.5초 지연).
            
        - **V3 데이터셋 반영 결과**
            
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
- 2013~2014년의 특정 오류 배치 89대 개체 제거 (마지막 데이터가 2014-02-13 이기 때문에 후처리로 전부 삭제됨)

<aside>

여전히 결측치가 남아있다면 분기 조건으로 사용하도록 놔둠

</aside>

- 데이터 컬럼
    
    
    | **이름** | **설명** | **비고** |
    | --- | --- | --- |
    | **smart_5_raw** | 불량 섹터 대체 횟수 |  |
    | **smart_184_raw** | 데이터 전송 경로 오류 |  |
    | **smart_187_raw** | 복구 불가 오류 횟수 |  |
    | **smart_197_raw** | 불안정 섹터 수 |  |
    | **smart_198_raw** | 복구 불가 섹터 수 |  |
    | **Timeout_5s** | 5초 초과 응답 지연 횟수 | 188에서 파생됨 |
    | **Timeout_7_5s** | 7.5초 초과 응답 지연 횟수 | 188에서 파생됨 |
    | **Timeout_Total** | 전체 응답 지연 횟수 | 188에서 파생됨 |
    | **seek_error_count** | 보이스 코일 액추에이터 탐색 오류 | 7에서 파생됨 |
    | **smart_9_raw** | 누적 사용 시간 |  |
    | **smart_189_raw** | 헤드 정렬 불량 쓰기 횟수 |  |
    | **smart_191_raw** | 외부 충격 감지 횟수 |  |
    | **smart_194_raw** | 드라이브 현재 온도 | 100도 이상은 ffill 적용 |
    | **smart_199_raw** | 케이블/통신 연결 오류 수 |  |
    | **smart_241_raw** | 누적 데이터 쓰기량 | 값 매우 큼 |
    | **smart_242_raw** | 누적 데이터 읽기량 | 값 매우 큼 |
    | **Total_Reads** | 총 읽기 섹터 시도 횟수 | 1에서 파생됨 |
    | **total_seeks** | 총 탐색 작업 횟수 | 7에서 파생됨 |
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
        - 역할: 튜닝에 오염되지 않은 독립된 셋. 오직 `목표 오탐률 n% 제약과 Recall 최대화의 최적 조합`을 달성하는 최적의 임계값을 정밀하게 도출하여, 실무 환경 도입 시 오탐 폭발을 원천 차단.
        
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
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_group_stratified\test_raw.parquet
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_group_stratified\train_raw.parquet
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_group_stratified\val_calib_raw.parquet
    저장 완료: C:\Workspace\06_ML_projdect\26_1_COIN\data\split_group_stratified\val_tune_raw.parquet
    ```
    

---

## 4. 특성 생성

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

---

## **5. 특성 선택**

#### fs_sample_train, fs_sample_test 파일 생성

```markdown
# RFE용 데이터셋 제작
1. 고장 개체 비율 fs_sample_train 8 : fs_sample_test 2
2. serial_number 단위에서 failure 비율 유지하여 분할
3. 학습 세트는 정상 행은 배정받은 개체 내부에서 랜덤시드를 사용하여 고장 행의 10배수 샘플링
4. 테스트 세트는 정상 행은 배정받은 개체 내부에서 랜덤시드를 사용하여 고장 행의 100배수 샘플링
    - 원본 불균형 1 : 1405.8이지만 타협한 수치
- (같은 serial_number는 train과 test에 동시에 존재하면 안 됨)
- seed = 42
```

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

### 3. 특성 선택

- 진행중
- 특성 계층 분석
</aside>

---

## **6. 하이퍼파라미터 최적화**

최종 선택된 변수를 반영한 데이터셋을 제작

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
        
    - 우리는 언더배깅을 기반으로 두고 샘플링 전략을 커스텀한 방식에 가까움.
        - 논문에서는 언더배깅 앙상블이라고 명명하고 구현 전략을 구체적으로 설명하는 쪽이 나을듯
    </aside>
    

<aside>

### 사용된 데이터 설명

full train: 학습을 위한 10개의 서브셋을 만들고 더 이상 사용되지 않음.

- 10개의 Subset:  앙상블 학습을 위한 경량 데이터셋
    - 모든 고장 데이터 + 10배수로 추출된 정상 데이터(near-failure에 샘플링 가중치 부여)

full validation: Optuna 루프에서는 미사용, 최종 선택 (rerank), 최종 검증 과정에서 사용됨

- sampled validation: 옵튜나 내부에서 사용되는 경량 데이터셋
    - 모든 고장 데이터 + 100배수로 추출된 정상 데이터(near-failure에 샘플링 가중치 부여)
</aside>

### [Optuna Stage 1]

- 대략적인 범위를 넓게 탐색함.
- 튜닝하기 까다로운 n_estimators를 400으로 고정하고 진행함.

```markdown
1. 10개의 subset을 순차적으로 학습함

2. 각 subset 학습 직후:
    - sampled validation을 예측하고
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

### [Optuna Stage 2]

- [Optuna Stage 1]의 결과를 바탕으로 좋은 영역의 주변을 정밀하게 탐색
- n_estimators 해방
- warm-start enqueue → 좋은 지점부터 시작
- pruning 비활성화 → 이미 좋은 후보군만 탐색하기 때문에 가지치기가 방해가 될 수 있음

```markdown
1. [Optuna Stage 1] best parameter를 첫 trial로 주입

2. 좁혀진 탐색 공간에서 추가 탐색 수행

3. 모든 trial 완주 후 sampled validation 기준 점수를 정렬
```

---

### [Reranking]

- sampled validation을 사용했을 때의 리스크를 최소화하기 위한 재평가 단계
- full validation 기준 최종 선택

```markdown
1. [Optuna Stage 2]의 결과 중 best parameter이 될 수 있는 후보 선택 (마진? 분포 확인? 미정)

2. 저장된 ensemble 모델 재사용 (없으면 학습)

3. full validation을 예측하여 PR-AUC 계산

4. 이를 기준으로 최종 하이퍼파라미터 선택
```

---

### [Final Training]

- 최종 선택된 하이퍼파라미터로 전체 앙상블 재학습 수행
- full validation 기준 최종 성능 확인
- 모델 및 파라미터 저장
- 탐색 공간
    
    
    | **Parameter** | **Type** | **Range** | **Log Scale** | **Rationale (설계 논리)** |
    | --- | --- | --- | --- | --- |
    | **`learning_rate`** | Float | `0.01 ~ 0.1` | **Yes** | 트리 개수(`n_estimators=400`)가 고정된 상태에서 수렴 속도를 맞추기 위해 넓은 로그 스케일로 탐색합니다. |
    | **`max_depth`** | Int | `4 ~ 10` | No | 과적합 방지를 위한 1차 방어선입니다. SMART 데이터의 노이즈 특성상 10을 초과하는 깊이는 유의미한 패턴보다 노이즈를 외울 확률이 높습니다. |
    | **`num_leaves`** | Int | `16 ~ 128` | No | 과적합 방지를 위한 2차 방어선입니다. 무의미한 탐색(예: depth=4인데 leaves=100)을 막기 위해, 실제 코드에서는 `min(2^max_depth, 128)`로 **동적 상한(Conditional Bound)**을 걸어 공간 낭비를 제거했습니다. |
    | **`min_child_samples`** | Int | `20 ~ 100` | No | 불균형 데이터에서 리프 노드가 극소수의 Positive 샘플에 과적합되는 것을 막는 최소한의 허들입니다. |
    | **`feature_fraction`** | Float | `0.5 ~ 1.0` | No | **(핵심)** Correlated SMART Feature 환경에서 트리가 특정 강한 노이즈 피처에 종속되는 것을 막고, 각 트리의 Subspace Sampling을 강화하여 **앙상블(Tree-level) 다양성을 증가시킴**. 단, 부스팅 과정 파괴를 막기 위해 마지노선(0.5)을 방어합니다. |
    | **`bagging_fraction`** | Float | `0.6 ~ 1.0` | No | 이미 외부에서 10:1 Underbagging 앙상블이 적용되어 있으므로, 내부의 Row-sampling은 중간 수준(0.6) 이상으로 유지하여 학습 데이터 손실을 막습니다. |
    | **`lambda_l1`** | Float | `1e-4 ~ 1.0` | **Yes** | L1 규제. 불필요한 split 사용을 억제하여 noisy feature 의존을 완화 |
    | **`lambda_l2`** | Float | `1e-8 ~ 10.0` | **Yes** | L2 규제. 잎사귀(Leaf)의 출력값을 부드럽게 눌러주어 leaf output의 과도한 진폭을 완화하는 방향으로 작용 |
    | **`bagging_freq`** | Int | **`1` (고정)** | - | Underbagging 체제이므로 항상 bagging이 켜져 있는 상태를 유지합니다. |
    | **`n_estimators`** | Int | **`400 ~ 800` (1차에서는 400으로 고정)** | - | Stage 1 에서는 빠른 파라미터 영역 필터링(Coarse Screening)과 파라미터 간 n_estimators 변동에 따른 탐색 노이즈를 줄이기 위해 고정 |

---

### 데이터 샘플링 심화

### 학습 데이터 샘플링

전체 학습 데이터(train.parquet)는 연산 효율성과 앙상블 다양성 확보를 위해
10개의 서로 다른 언더샘플링 서브셋으로 분할하였다.

- 모든 failure sample은 각 서브셋에 100% 포함
- normal sample은 비복원 추출 기반으로 10:1 비율 유지
- 각 서브셋 구성
    - failure: 33,984
    - normal: 339,840
    - total: 373,824
- 전체 ensemble
    - 10 subsets → 총 3,738,240 samples

---

### Near-failure Importance Sampling

고장 직전 구간의 정보 손실을 줄이기 위해
time-to-failure 기반 importance sampling을 적용하였다.

- failure 이전 일정 구간
    - failure: D-1 ~ D-10
    - near-failure: D-11 ~ D-30
- 해당 구간의 샘플링 확률을 3배 증가

이를 통해 정상 상태와 명확히 구분되지 않는   

고장 직전 패턴이 학습 과정에서 충분히 반영되도록 하였다.

> **인용 포인트:** *Learning from Imbalanced Data* (He & Garcia, 2009) 논문을 보면, "결정 경계(Decision Boundary) 근처에 있는 다수 클래스 샘플(Borderline majority examples)을 보존하는 것이 모델의 판별 성능 향상에 기여할 수 있다고 보고하였다."고 되어 있습니다.
> 

---

### 검증 데이터 샘플링

전체 검증 데이터에 대한 반복 평가 과정은
하이퍼파라미터 탐색 시 주요 연산 병목으로 작용하였다.

따라서 Optuna 기반 튜닝 단계에서는
검증셋에 대해 별도의 언더샘플링을 적용하였다.

튜닝 단계의 목적은 절대적인 성능 추정이 아니라
모델 간 상대적 성능 비교의 안정적 수행에 있으므로,
실제 분포를 일부 압축한 검증셋을 사용하는 것은 타당하다고 판단하였다.

- 모든 failure sample 유지
- normal sample만 추가 샘플링 수행
- 최종 검증 비율은 약 100:1 유지
- near-failure 구간에 대해서는 학습 데이터와 동일하게 3배 importance sampling 적용

---

### (참고) 언더샘플링의 근거

정상 클래스는 다음 특성을 가진다:

- 높은 temporal redundancy
- 장기간 안정 상태 반복

따라서 전체 정상 데이터를 모두 사용하는 대신:

- 행(row) 단위 random undersampling을 통해
- 정보량이 높은 다양한 정상 상태를 유지하면서
- 계산 비용을 줄인다

이로 인해:

- redundancy 감소
- ensemble diversity 증가
- memory constraint 해결

---

### 성능 비교

검증셋의 불균형이 다르기 때문에 직접적인 PR-AUC 비교는 불가능함.

| 튜닝 전 | 노트북 학습 & 검증 시간 | 데스크탑 학습 & 검증 시간 | PR-AUC |
| --- | --- | --- | --- |
| full validation (1:1426.4 불균형) | 2~30분 예상 | 3m 28.6s | 0.11249 |
| sampled validation (100:1 불균형) | 8m 8.8s | 1m 11.4s | 0.40907  |

| 튜닝 후 | PR-AUC | std |
| --- | --- | --- |
| full validation (1:1426.4 불균형) |  |  |
| sampled validation (100:1 불균형) |  |  |

---

## 7. 임계값 튜닝

제약 조건 최적화

<aside>

1. val_calib.parquet로 임계점 그리드서치
2. FPR과 Recall 의 관계를 그래프로 그림 (x: FPR, y: Recall(TPR))
3. 적절한 지점을 도출함

FPR n%에서 Recall nn%라는 문장을 뽑아야 함. (우리 연구의 간판)

→ 허용한 오탐율 내에서 탐지 성능을 최대화한 운영점

</aside>

<aside>

| FPR | Recall |
| --- | --- |
| 0.1% | xx% |
| 0.5% | xx% |
| 1.0% | xx% |
| 5.0% | xx% |
</aside>

---

## 8. 모델 평가

## **1. 행 단위 점수 평가**

모델이 각 시점(행)에 대해 출력한 확률을 기준으로, threshold를 적용하여 성능을 정량적으로 평가한다.

- **평가 목적**
    - 모델 자체의 예측 품질 및 분류 능력 평가
    - 서로 다른 모델/특성 조합 비교
- **평가 지표**
    - PR-AUC (ranking 성능)
    - MCC (class imbalance 반영)
    - F1-score (threshold 기반 성능)
- **특징**
    - 모든 샘플을 독립적으로 평가
    - 시간 구조는 고려하지 않음

---

## **2. 실무형 롤링 평가**

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
    - “얼마나 빨리, 얼마나 정확하게 잡는가” 평가

---

## 9. 고장 해석

### **8.1 전역적 해석 (Global Interpretability)**

SHAP Summary Plot을 기반으로 전체 데이터에서 주요 고장 기여 변수를 분석한다.

- 주요 SMART feature 중요도 분석
- 전반적인 고장 패턴 구조 파악

---

### **8.2 국소적 해석 (Local Interpretability)**

개별 디스크 단위에서 SHAP Waterfall Plot을 활용하여 특정 예측 결과의 근거를 설명한다.

- 시리얼 넘버별 고장 원인 분석
- 엔지니어 해석 가능 형태 제공

---

### **8.3 시간 기반 해석 (Temporal Interpretation)**

고장은 시간에 따른 점진적 열화 과정이므로, 일정 기간(window)의 feature 변화가 예측에 미치는 영향을 분석한다.

- 고장 전 일정 기간(예: 30일) feature trajectory 분석
- SHAP 기반 기여도 변화 추적

---

### **8.4 실무 활용 효과**

- 조기 경보 원인 설명 가능
- 유지보수 판단 근거 제공
- 모델 신뢰성 향상