is_warmup_28d	최근 부하 압력	$\mathbb{I}(t_{elapsed} < 28)$	28일 관측 구간 부족 플래그	초기 관측 구간(노이즈) 필터링
is_warmup_14d	최근 부하 압력	$\mathbb{I}(t_{elapsed} < 14)$	14일 관측 구간 부족 플래그	초기 관측 구간(노이즈) 필터링
is_warmup_7d	최근 부하 압력	$\mathbb{I}(t_{elapsed} < 7)$	7일 관측 구간 부족 플래그	초기 관측 구간(노이즈) 필터링
workload_intensity	최근 부하 압력	$\frac{s9_t + 1}{\Delta s241_t + \Delta s242_t + 1}$	누적 사용 시간 대비 총 작업량 비율	나이(누적치) 대비 당일 총 작업량 비율
age_weighted_workload	최근 부하 압력	$\ln(\lvert\Delta s241_t + \Delta s242_t\rvert + 1) \times \ln(s9_t + 1)$	노후도 기반 일일 작업 부하 가중치	작업 감소(음수) 시 에러 방지용 절댓값 적용
workload_7d_accel	최근 부하 압력	$(\Delta s241_t + \Delta s242_t) - (\Delta s241_{t-7} + \Delta s242_{t-7})$	총 작업량 7일 가속도	Temporal Diff (유지)
total_seeks_28d_ewma	총 탐색량	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	총 탐색 작업 28일 지수가중	28일 장기적 기계 탐색 트렌드를 완만히 따라가는 이동평균 지표
total_seeks_28d_dai	총 탐색량	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	총 탐색 작업 28일 평균증가량	28일 만성적 추세에서 보여지는 일평균 탐색 증감 기울기
total_seeks_28d_asfd	총 탐색량	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 탐색 작업 28일 차분절대합	28일간 잦은 탐색 횟수 변동으로 쌓인 만성적 물리 구동 피로합
total_seeks_28d_zscore	총 탐색량	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	총 탐색 작업 28일 Z스코어	28일 만성 기저 평균을 넘어선 당일의 헤드 탐색 비정상 수치
total_seeks_28d_std	총 탐색량	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	총 탐색 작업 28일 차분 편차	28일 장기 만성적인 기계 헤드 동작 횟수의 산포 불규칙성
total_seeks_28d_mean	총 탐색량	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	총 탐색 작업 28일 차분 평균	28일 장기간에 걸쳐 평균화된 만성 기계 탐색 기저 부하
total_seeks_28d_sum	총 탐색량	$\sum_{i=0}^{27} \Delta W_{t-i}$	총 탐색 작업 28일 차분 총합	28일 만성 장기 구동 과정에서 누적된 기계 탐색 부하 총량
total_seeks_28d_max	총 탐색량	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	총 탐색 작업 28일 차분 최댓값	28일 내 헤드 구동 기계 부하가 집중된 장기 최대 스파이크 일
total_seeks_14d_ewma	총 탐색량	$0.133 \cdot \Delta W_t + 0.867 \cdot E_{t-1}$	총 탐색 작업 14일 지수가중	14일 중기 탐색 작업 패턴을 부드럽게 지시하는 트렌드 지표
total_seeks_14d_dai	총 탐색량	$\frac{\Delta W_t - \Delta W_{t-13}}{13}$	총 탐색 작업 14일 평균증가량	14일 중기 탐색 횟수의 전반적 증감 추세 기울기
total_seeks_14d_asfd	총 탐색량	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 탐색 작업 14일 차분절대합	14일 중기 탐색 횟수의 지속적인 널뛰기로 인한 변동 피로도
total_seeks_14d_zscore	총 탐색량	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	총 탐색 작업 14일 Z스코어	14일 중기 기저 상태를 반영한 당일 탐색 횟수 이상 스케일
total_seeks_14d_std	총 탐색량	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	총 탐색 작업 14일 차분 편차	14일 중기 탐색 동작의 요동치는 분산 수준
total_seeks_14d_mean	총 탐색량	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	총 탐색 작업 14일 차분 평균	14일간 중기 일평균 헤드 탐색 기계 동작 횟수
total_seeks_14d_sum	총 탐색량	$\sum_{i=0}^{13} \Delta W_{t-i}$	총 탐색 작업 14일 차분 총합	14일 중기적으로 누적된 헤드 탐색 이동 부하 데미지
total_seeks_14d_max	총 탐색량	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	총 탐색 작업 14일 차분 최댓값	14일 내 헤드 구동 액추에이터의 발생 최대 트래픽 스파이크
total_seeks_7d_ewma	총 탐색량	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	총 탐색 작업 7일 지수가중	7일 내 탐색 트래픽 추세에 지수적으로 민감하게 반응하는 평균
total_seeks_7d_dai	총 탐색량	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	총 탐색 작업 7일 평균증가량	7일간 매일 기계 탐색 작업이 늘거나 줄어든 선형 추세
total_seeks_7d_asfd	총 탐색량	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 탐색 작업 7일 차분절대합	7일간 헤드 탐색 횟수의 급증/급감 누적 플래핑(Flapping)
total_seeks_7d_zscore	총 탐색량	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	총 탐색 작업 7일 Z스코어	과거 7일 탐색 부하 대비 당일 탐색 시도의 스파이크 지수
total_seeks_7d_std	총 탐색량	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	총 탐색 작업 7일 차분 편차	7일간 헤드 탐색 작업의 불규칙한 산포도
total_seeks_7d_mean	총 탐색량	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	총 탐색 작업 7일 차분 평균	7일간 단기 일평균 헤드 탐색 구동 부하 빈도
total_seeks_7d_sum	총 탐색량	$\sum_{i=0}^{6} \Delta W_{t-i}$	총 탐색 작업 7일 차분 총합	7일간 액추에이터 기계 부품이 수행한 탐색 작업 총량
total_seeks_7d_max	총 탐색량	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	총 탐색 작업 7일 차분 최댓값	7일 내 디스크 헤드가 트랙을 가장 많이 찾아다닌 일일 피크
total_reads_28d_ewma	총 읽기량	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	총 읽기 시도 28일 지수가중	28일 장기적 동작 패턴 트렌드를 부드럽게 추종하는 추세
total_reads_28d_dai	총 읽기량	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	총 읽기 시도 28일 평균증가량	28일 장기간에 걸친 평균적 동작 빈도 증감 추세 기울기
total_reads_28d_asfd	총 읽기량	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 읽기 시도 28일 차분절대합	28일간 동작 빈도 널뛰기로 인해 축적된 장기 피로합
total_reads_28d_zscore	총 읽기량	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	총 읽기 시도 28일 Z스코어	28일 장기 평균 기저 상태 대비 당일 읽기 동작 이례성
total_reads_28d_std	총 읽기량	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	총 읽기 시도 28일 차분 편차	28일 장기 읽기 동작 횟수의 불규칙 분산 폭
total_reads_28d_mean	총 읽기량	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	총 읽기 시도 28일 차분 평균	28일 장기간 평균적으로 감당해온 만성 동작 부하
total_reads_28d_sum	총 읽기량	$\sum_{i=0}^{27} \Delta W_{t-i}$	총 읽기 시도 28일 차분 총합	28일 동안 드라이브 구동부에 가해진 장기 누적 동작 수
total_reads_28d_max	총 읽기량	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	총 읽기 시도 28일 차분 최댓값	28일 내 가장 집중적으로 기계가 동작한 날의 횟수 스파이크
total_reads_14d_ewma	총 읽기량	$0.133 \cdot \Delta W_t + 0.867 \cdot E_{t-1}$	총 읽기 시도 14일 지수가중	14일간 읽기 횟수 증감 트렌드의 지수가중 이동평균
total_reads_14d_dai	총 읽기량	$\frac{\Delta W_t - \Delta W_{t-13}}{13}$	총 읽기 시도 14일 평균증가량	14일 중기 일일 동작 횟수 선형 추이 기울기
total_reads_14d_asfd	총 읽기량	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 읽기 시도 14일 차분절대합	14일 중기 읽기 동작 횟수의 급변 피로도 거리
total_reads_14d_zscore	총 읽기량	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	총 읽기 시도 14일 Z스코어	14일 중기 동작 기준치 대비 당일 작업 폭주 스케일
total_reads_14d_std	총 읽기량	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	총 읽기 시도 14일 차분 편차	14일간 읽기 동작 빈도의 중기적 불규칙 산포도
total_reads_14d_mean	총 읽기량	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	총 읽기 시도 14일 차분 평균	14일간 디스크 구동부의 중기 일평균 동작 부하 빈도
total_reads_14d_sum	총 읽기량	$\sum_{i=0}^{13} \Delta W_{t-i}$	총 읽기 시도 14일 차분 총합	14일간 가해진 중기 누적 순수 읽기 횟수 총합
total_reads_14d_max	총 읽기량	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	총 읽기 시도 14일 차분 최댓값	14일 내 발생한 하루 기준 읽기 동작 횟수의 최대 피크
total_reads_7d_ewma	총 읽기량	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	총 읽기 시도 7일 지수가중	7일 내 읽기 동작 빈도 증가 트렌드의 단기 지수가중
total_reads_7d_dai	총 읽기량	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	총 읽기 시도 7일 평균증가량	7일간 단기 일일 동작 횟수 증감 추세선 기울기
total_reads_7d_asfd	총 읽기량	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	총 읽기 시도 7일 차분절대합	7일간 동작 횟수 증가/감소가 누적된 총 굴곡 거리
total_reads_7d_zscore	총 읽기량	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	총 읽기 시도 7일 Z스코어	과거 7일 동작 평균 대비 당일 횟수 폭주 이상치
total_reads_7d_std	총 읽기량	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	총 읽기 시도 7일 차분 편차	7일간 읽기 동작 횟수의 기계적 불규칙 산포
total_reads_7d_mean	총 읽기량	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	총 읽기 시도 7일 차분 평균	7일간 일평균 디스크 읽기 동작 횟수
total_reads_7d_sum	총 읽기량	$\sum_{i=0}^{6} \Delta W_{t-i}$	총 읽기 시도 7일 차분 총합	7일간 드라이브에 가해진 순수 읽기 동작 실행 총 횟수
total_reads_7d_max	총 읽기량	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	총 읽기 시도 7일 차분 최댓값	7일 내 하루 기준 디스크 암이 읽기 동작을 수행한 횟수 피크
s242_28d_ewma	총 읽기량	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	누적 읽기량 28일 지수가중	28일 장기 읽기 부하의 부드러운 지수가중 트렌드
s242_28d_dai	총 읽기량	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	누적 읽기량 28일 평균증가량	28일 장기간 읽기 트래픽의 선형 추세 기울기
s242_28d_asfd	총 읽기량	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 읽기량 28일 차분절대합	28일 읽기 트래픽의 변화 총량 및 장기 굴곡 피로도
s242_28d_zscore	총 읽기량	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	누적 읽기량 28일 Z스코어	28일 장기 기저 상태 대비 당일 읽기 폭주 감지
s242_28d_std	총 읽기량	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	누적 읽기량 28일 차분 편차	28일간 만성적인 장기 읽기 부하의 불안정성
s242_28d_mean	총 읽기량	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	누적 읽기량 28일 차분 평균	28일간 디스크가 지속 감당한 기저 읽기 부하 상태
s242_28d_sum	총 읽기량	$\sum_{i=0}^{27} \Delta W_{t-i}$	누적 읽기량 28일 차분 총합	28일 동안 기록된 만성 장기 총 읽기 데이터량
s242_28d_max	총 읽기량	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	누적 읽기량 28일 차분 최댓값	28일 내 발생한 하루 최대 읽기 데이터 스파이크
s242_14d_ewma	총 읽기량	$0.133 \cdot \Delta W_t + 0.867 \cdot E_{t-1}$	누적 읽기량 14일 지수가중	최근 14일 내 읽기량 트렌드 민감 지수가중 평균
s242_14d_dai	총 읽기량	$\frac{\Delta W_t - \Delta W_{t-13}}{13}$	누적 읽기량 14일 평균증가량	14일간 중기 일평균 읽기 부하 선형 트렌드
s242_14d_asfd	총 읽기량	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 읽기량 14일 차분절대합	14일간 잦은 읽기 요청 증감의 피로도 누적
s242_14d_zscore	총 읽기량	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	누적 읽기량 14일 Z스코어	과거 14일 기준 대비 당일 읽기 부하 이상치 척도
s242_14d_std	총 읽기량	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	누적 읽기량 14일 차분 편차	14일간 읽기 부하의 중기적 불규칙 산포도
s242_14d_mean	총 읽기량	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	누적 읽기량 14일 차분 평균	14일간 디스크가 처리한 중기 일평균 읽기 부하량
s242_14d_sum	총 읽기량	$\sum_{i=0}^{13} \Delta W_{t-i}$	누적 읽기량 14일 차분 총합	14일 동안 기록된 순수 중기 총 읽기 데이터량
s242_14d_max	총 읽기량	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	누적 읽기량 14일 차분 최댓값	14일 내 발생한 하루 최대 데이터 읽기 부하 피크
s242_7d_ewma	총 읽기량	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	누적 읽기량 7일 지수가중	최근 7일 내 단기 읽기 부하 폭증 감지 추세
s242_7d_dai	총 읽기량	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	누적 읽기량 7일 평균증가량	7일간 일평균 단기 읽기 부하 상승 선형 추세
s242_7d_asfd	총 읽기량	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 읽기량 7일 차분절대합	7일간 읽기 요청 급증/급감을 반복한 총 이동 거리
s242_7d_zscore	총 읽기량	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	누적 읽기량 7일 Z스코어	과거 7일 읽기량 대비 당일 폭주 수준 스케일링
s242_7d_std	총 읽기량	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	누적 읽기량 7일 차분 편차	7일간 읽기 부하의 불규칙한 산포도
s242_7d_mean	총 읽기량	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	누적 읽기량 7일 차분 평균	7일간 디스크가 처리한 단기 일평균 읽기 부하량
s242_7d_sum	총 읽기량	$\sum_{i=0}^{6} \Delta W_{t-i}$	누적 읽기량 7일 차분 총합	7일 동안 드라이브가 순수하게 읽어낸 총 데이터량
s242_7d_max	총 읽기량	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	누적 읽기량 7일 차분 최댓값	7일 내 발생한 하루 최대 데이터 읽기 부하 피크
s242_7d_accel	총 읽기량			
s242_diff	총 읽기량	$\Delta s242_t$	읽기량 1일 변화량	일일 읽기 강도
smart_242_raw	총 읽기량		Total LBAs Read 원본 값	누적 읽기량 (상태)
s241_28d_ewma	총 기록량	$0.069 \cdot \Delta W_t + 0.931 \cdot E_{t-1}$	누적 쓰기량 28일 지수가중	장기 28일 쓰기량 트렌드 파악 지수가중 이동평균
s241_28d_dai	총 기록량	$\frac{\Delta W_t - \Delta W_{t-27}}{27}$	누적 쓰기량 28일 평균증가량	28일간 만성적인 일일 쓰기 트래픽 증가/감소 추세
s241_28d_asfd	총 기록량	$\sum_{i=0}^{27} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 쓰기량 28일 차분절대합	28일간 잦은 I/O 전환으로 누적된 장기 부하 피로도
s241_28d_zscore	총 기록량	$\frac{\Delta W_t - \mu_{28}}{\sigma_{28} + 1e-5}$	누적 쓰기량 28일 Z스코어	장기 28일 기준치 대비 당일 부하의 이상 수치
s241_28d_std	총 기록량	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(\Delta W_{t-i} - \mu_{28})^2}$	누적 쓰기량 28일 차분 편차	28일간 쓰기 부하의 장기 만성적 불규칙 수준
s241_28d_mean	총 기록량	$\frac{1}{28}\sum_{i=0}^{27} \Delta W_{t-i}$	누적 쓰기량 28일 차분 평균	28일간 디스크가 장기적으로 감당한 기저 쓰기 부하
s241_28d_sum	총 기록량	$\sum_{i=0}^{27} \Delta W_{t-i}$	누적 쓰기량 28일 차분 총합	28일 동안 기록된 만성적인 순수 쓰기 총 데이터량
s241_28d_max	총 기록량	$\max(\Delta W_{t-27}, \dots, \Delta W_t)$	누적 쓰기량 28일 차분 최댓값	28일 내 발생한 하루 최대 쓰기 데이터 스파이크
s241_14d_asfd	총 기록량	$\sum_{i=0}^{13} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 쓰기량 14일 차분절대합	14일간 중기적 쓰기 부하의 전환 피로도 거리
s241_14d_zscore	총 기록량	$\frac{\Delta W_t - \mu_{14}}{\sigma_{14} + 1e-5}$	누적 쓰기량 14일 Z스코어	과거 14일 평균 대비 당일 쓰기량 폭주 통계적 척도
s241_14d_std	총 기록량	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(\Delta W_{t-i} - \mu_{14})^2}$	누적 쓰기량 14일 차분 편차	14일간 쓰기 부하량의 중기적 불규칙 산포도
s241_14d_mean	총 기록량	$\frac{1}{14}\sum_{i=0}^{13} \Delta W_{t-i}$	누적 쓰기량 14일 차분 평균	14일간 디스크가 처리한 중기 일평균 쓰기 부하량
s241_14d_sum	총 기록량	$\sum_{i=0}^{13} \Delta W_{t-i}$	누적 쓰기량 14일 차분 총합	14일 동안 기록된 순수 쓰기 데이터 중기 총량
s241_14d_max	총 기록량	$\max(\Delta W_{t-13}, \dots, \Delta W_t)$	누적 쓰기량 14일 차분 최댓값	14일 내 발생한 하루 최대 데이터 쓰기 부하 피크
s241_7d_ewma	총 기록량	$0.25 \cdot \Delta W_t + 0.75 \cdot E_{t-1}$	누적 쓰기량 7일 지수가중	최근 7일 쓰기량 폭증에 가중치를 둔 트렌드
s241_7d_dai	총 기록량	$\frac{\Delta W_t - \Delta W_{t-6}}{6}$	누적 쓰기량 7일 평균증가량	7일간 일평균 쓰기 부하 상승 선형 추세
s241_7d_asfd	총 기록량	$\sum_{i=0}^{6} \lvert \Delta W_{t-i} - \Delta W_{t-i-1} \rvert$	누적 쓰기량 7일 차분절대합	7일간 쓰기 부하의 급증/급감을 반복한 총 이동 거리
s241_7d_zscore	총 기록량	$\frac{\Delta W_t - \mu_7}{\sigma_7 + 1e-5}$	누적 쓰기량 7일 Z스코어	과거 7일 평균 대비 당일 쓰기량 폭주의 이상 수치
s241_7d_std	총 기록량	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(\Delta W_{t-i} - \mu_{7})^2}$	누적 쓰기량 7일 차분 편차	7일간 쓰기 부하량의 불규칙한 요동(산포도)
multi_error_count	Burst 이상			
s241_7d_mean	총 기록량	$\frac{1}{7}\sum_{i=0}^{6} \Delta W_{t-i}$	누적 쓰기량 7일 차분 평균	7일간 디스크가 처리한 일평균 쓰기 부하량
s5_daily_failure_speed	직접 손상 발생			
recovery_failure_flag	지속 악화			
cascading_failure_flag	시스템성 실패			
s241_7d_sum	총 기록량	$\sum_{i=0}^{6} \Delta W_{t-i}$	누적 쓰기량 7일 차분 총합	7일 동안 기록된 순수 쓰기 데이터 총량
s241_7d_max	총 기록량	$\max(\Delta W_{t-6}, \dots, \Delta W_t)$	누적 쓰기량 7일 차분 최댓값	7일 내 발생한 하루 최대 데이터 쓰기 부하 피크
s241_diff	총 기록량	$\Delta s241_t$	쓰기량 1일 변화량	일일 쓰기 강도
smart_241_raw	총 기록량		Total LBAs Written 원본 값	누적 쓰기량 (상태)
smart_9_raw	누적 사용량		누적 사용 시간	장기 aging
s5_relative_score_14d	Burst 이상,Reallocated / Pending	$\frac{s5_t}{P_{95}(s5_{t-14:t-1}) + 1}$	14일 최고점 대비 당일 불량 섹터 비율	[이상치 방어] $\max$가 유발하는 정규화 왜곡을 막기 위해 $P_{95}$ 백분위수로 대체
s192_14d_burst	Burst 이상	$\sum_{i=0}^{13} \Delta s192_{t-i}$	14일 강제 종료 단기 폭주량	[이름 수정] Density $\rightarrow$ Burst로 의미 일치
s189_28d_highfly_burst	Burst 이상	$\sum_{i=0}^{27} \Delta s189_{t-i}$	28일간 불량 쓰기 증가량	단순 변화량 윈도우 합산 (유지)
s187_14d_burst_index	Burst 이상	$\sum_{i=0}^{13} \left( \Delta s187_{t-i} \cdot \mathbb{I}(\Delta s187_{t-i} > 0) \right)$	복구 불가 오류 횟수 중기 누적 증가량	증가된(양수) 부분만 필터링 합산 (유지)
read_spike_ratio	Burst 이상,급성 Spike	$\frac{\Delta s242_t}{\left( \frac{1}{7}\sum_{i=1}^{7} \Delta s242_{t-i} \right) + 1}$	7일 평균 대비 당일 읽기량 폭주 비율	당일($t$) 제외 과거 7일 기준 (유지)
s189_28d_sum	Burst 이상	$\sum_{i=0}^{27} \Delta E_{t-i}$	헤드 정렬 불량 쓰기 28일 총합	28일 내 발생한 기계적 구동 불안정성 누적 스트레스
s189_28d_max	Burst 이상	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	헤드 정렬 불량 쓰기 28일 최댓값	28일 내 발생한 헤드 정렬 불량 쓰기의 최대 피크
s187_14d_sum	Burst 이상,Sector 열화	$\sum_{i=0}^{13} \Delta E_{t-i}$	복구 불가 오류 14일 총합	14일 내 발생한 치명적 복구 불가 에러 누적량
s187_14d_max	Burst 이상,Sector 열화	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	복구 불가 오류 14일 최댓값	14일 내 발생한 복구 불가(Uncorrectable) 에러 피크
s189_diff	Burst 이상	$\Delta s189_t$	write/seek burst 변화량	급격한 헤드 부상 이상 탐지
smart_189_raw	Burst 이상		write/seek burst 원본 값	기계적 결함 상태
write_spike_ratio	급성 Spike	$\frac{\Delta s241_t}{\left( \frac{1}{7}\sum_{i=1}^{7} \Delta s241_{t-i} \right) + 1}$	7일 평균 대비 당일 쓰기량 폭주 비율	$\Delta$ 기반 폭주 연산 (유지)
uncorrectable_spike_ratio	급성 Spike	$\frac{\Delta s198_t}{\left( \frac{1}{14}\sum_{i=1}^{14} \Delta s198_{t-i} \right) + 1}$	복구 불가 섹터 중기 폭증 비율	$\Delta$ 기반 폭주 연산 (유지)
seek_spike_ratio	급성 Spike	$\frac{\Delta Total\_Seeks_t}{\left( \frac{1}{7}\sum_{i=1}^{7} \Delta Total\_Seeks_{t-i} \right) + 1}$	7일 평균 대비 당일 탐색 폭주 비율	$\Delta$ 기반 폭주 연산 (유지)
write_stability_ratio	읽기/쓰기 안정성	$\frac{s189_t + 1}{\Delta s241_t + 1}$	쓰기량 대비 헤드 정렬 불량률	당일 쓰기량 대비 누적 헤드 정렬 불량 비율
s183_28d_sum	읽기/쓰기 안정성	$\sum_{i=0}^{27} \Delta E_{t-i}$	SATA 속도 저하 28일 총합	28일 내 SATA 속도 저하 장기 누적 발생 횟수
s183_28d_max	읽기/쓰기 안정성	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	SATA 속도 저하 28일 최댓값	28일 내 SATA 속도 저하 현상의 최대 피크
s183_14d_sum	읽기/쓰기 안정성	$\sum_{i=0}^{13} \Delta E_{t-i}$	SATA 속도 저하 14일 총합	14일 내 SATA 속도 저하 현상 누적 발생 횟수
s183_14d_max	읽기/쓰기 안정성	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	SATA 속도 저하 14일 최댓값	14일 내 SATA 속도 저하 현상의 최대 피크
s183_diff	읽기/쓰기 안정성	$\Delta s183_t$	SATA 속도 변화	degradation rate
smart_183_raw	읽기/쓰기 안정성		SATA 속도 저하	전송 품질 상태
age_weighted_seek_error	Seek 경로 이상	$\Delta Seek\_Error_t \times s9_t$	노후도 대비 탐색 오류 심각도	당일 발생량과 누적 사용 시간 곱셈
timeout_seek_density	Seek 경로 이상	$\frac{Timeout\_Total_t}{Total\_Seeks_t + 1}$	탐색 작업 대비 에러율 (밀도)	역방향 뒤집어 Density로 통일
seek_error_density	Seek 경로 이상	$\frac{Seek\_Error\_Count_t}{Total\_Seeks_t + 1}$	탐색당 탐색 오류 밀도	역방향 뒤집어 Density로 통일
seek_error_count_28d_sum	Seek 경로 이상	$\sum_{i=0}^{27} \Delta E_{t-i}$	탐색 오류 28일 총합	28일 내 기계 구동부 탐색 실패 장기 누적 횟수
seek_error_count_28d_max	Seek 경로 이상	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	탐색 오류 28일 최댓값	28일 내 액추에이터 탐색 실패 횟수의 최대 피크
seek_error_count_14d_sum	Seek 경로 이상	$\sum_{i=0}^{13} \Delta E_{t-i}$	탐색 오류 14일 총합	14일 내 기계 구동부 탐색 실패 누적 횟수
seek_error_count_14d_max	Seek 경로 이상	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	탐색 오류 14일 최댓값	14일 내 액추에이터 탐색 실패 횟수의 최대 피크
seek_error_count_diff	Seek 경로 이상	$\Delta Seek\_Error\_Count_t$	탐색 오류 변화량	헤드 불안정 속도
s199_days_since_last	기본 I/O 이상	$\begin{cases} t - \max\{\tau \mid \max(0, \Delta s199_\tau) > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	통신 오류 증가 이후 무사고 경과일	[표준 cases] 순수 증가 이벤트 기준
timeout_to_uncorrectable_lag1	기본 I/O 이상	$\max(0, \Delta Timeout\_Total_t) \times \max(0, \Delta s198_{t-1})$	시차 결합 지수	어제 불량 발생 $\times$ 오늘 지연 발생
timeout_severity_score	기본 I/O 이상	$\frac{Timeout\_Total_t + 1}{Timeout\_5s_t + 1}$	타임아웃 심각도 비율	전체 타임아웃 대비 치명적 지연(5s) 비율
timeout_read_density	기본 I/O 이상	$\frac{Timeout\_Total_t}{Total\_Reads_t + 1}$	읽기 작업 대비 에러율 (밀도)	역방향 뒤집어 Density로 통일
s199_error_density	기본 I/O 이상	$\frac{s199_t}{Total\_Reads_t + Total\_Seeks_t + 1}$	작업당 통신 연결 오류 밀도	역방향 뒤집어 Density로 통일
io_asymmetry_index	기본 I/O 이상	$\frac{\lvert\Delta s241_t - \Delta s242_t\rvert}{\Delta s241_t + \Delta s242_t + 1}$	읽기/쓰기 작업 비대칭 지수	작업량 차이의 절댓값을 전체로 나눔
s199_14d_burst	기본 I/O 이상	$\sum_{i=0}^{13} \Delta s199_{t-i}$	14일간 통신 오류 단기 폭주량	단순 변화량 윈도우 합산 (유지)
timeout_total_28d_sum	기본 I/O 이상	$\sum_{i=0}^{27} \Delta E_{t-i}$	전체 지연 28일 총합	28일 내 펌웨어 응답 지연 현상 장기 누적 횟수
timeout_total_28d_max	기본 I/O 이상	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	전체 지연 28일 최댓값	28일 내 발생한 타임아웃 횟수의 최대 피크
timeout_total_14d_sum	기본 I/O 이상	$\sum_{i=0}^{13} \Delta E_{t-i}$	전체 지연 14일 총합	14일 내 펌웨어 응답 지연 현상 누적 횟수
timeout_total_14d_max	기본 I/O 이상	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	전체 지연 14일 최댓값	14일 내 발생한 타임아웃 횟수의 최대 피크
timeout_5s_28d_sum	기본 I/O 이상	$\sum_{i=0}^{27} \Delta E_{t-i}$	5초 지연 28일 총합	28일 내 치명적 펌웨어 멈춤 장기 누적 발생 횟수
timeout_5s_28d_max	기본 I/O 이상	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	5초 지연 28일 최댓값	28일 내 발생한 5초 초과 I/O 펜딩의 최대 피크
timeout_5s_14d_sum	기본 I/O 이상	$\sum_{i=0}^{13} \Delta E_{t-i}$	5초 지연 14일 총합	14일 내 치명적 펌웨어 멈춤 현상의 누적 발생 횟수
timeout_5s_14d_max	기본 I/O 이상	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	5초 지연 14일 최댓값	14일 내 발생한 5초 초과 I/O 펜딩(멈춤)의 단기 피크
s199_28d_sum	기본 I/O 이상	$\sum_{i=0}^{27} \Delta E_{t-i}$	통신 오류 28일 총합	28일 내 통신 에러 누적 발생 장기 횟수 합산
s199_28d_max	기본 I/O 이상	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	통신 오류 28일 최댓값	28일 내 컨트롤러 간 통신 에러의 최대 피크
s199_14d_sum	기본 I/O 이상	$\sum_{i=0}^{13} \Delta E_{t-i}$	통신 오류 14일 총합	14일 내 통신 에러 누적 발생 횟수 합산
s199_14d_max	기본 I/O 이상	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	통신 오류 14일 최댓값	14일 내 컨트롤러 간 통신 에러(CRC)의 최대 피크
timeout_5s	기본 I/O 이상		5초 지연 원본 값	극단 지연 상태
timeout_5s_diff	기본 I/O 이상	$\Delta Timeout\_5s_t$	5초 지연 변화량	급성 freeze 탐지
timeout_total	기본 I/O 이상		전체 타임아웃 원본 값	누적 지연 상태
timeout_total_diff	기본 I/O 이상	$\Delta Timeout\_Total_t$	전체 타임아웃 변화량	시스템 악화 속도
s199_diff	기본 I/O 이상	$\Delta s199_t$	UDMA CRC Error 변화량	통신 불안정성 증가
smart_199_raw	기본 I/O 이상		UDMA CRC Error 원본 값	인터페이스 오류 누적
log_shock_fly_interaction	물리 스트레스 상호작용	$\ln(1 + \lvert\Delta s191_t \times \Delta s189_t\rvert) \times \text{sgn}(\Delta s191_t \times \Delta s189_t)$	외부 충격 및 헤드 비행 오류 동시 발생 심각도	로그 내 음수 충돌 방지 및 부호 복원
fatal_crash_interaction	물리 스트레스 상호작용	$\Delta Timeout\_5s_t \times \Delta s187_t$	응답 지연 및 복구 불가 오류 결합 지수	변화량 기준 단순 곱셈
shock_to_highfly_ratio	물리 스트레스 상호작용	$\frac{\sum_{i=0}^{27} \lvert\Delta s191_{t-i}\rvert + 1}{\sum_{i=0}^{27} \lvert\Delta s189_{t-i}\rvert + 1}$	충격 대비 헤드 불안정 전이 비율	[음수 상쇄 방어] 절댓값(\lvert, \rvert) 씌워서 순수 누적 충격량만 비교
s191_days_since_last	기계적 충격	$\begin{cases} t - \max\{\tau \mid \max(0, \Delta s191_\tau) > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	외부 충격 발생 이후 무사고 경과일	[표준 cases] 순수 증가 이벤트 기준
shock_seek_interaction	기계적 충격	$\Delta s191_t \times \Delta Seek\_Error_t$	충격 및 탐색 오류 동시 발생 결합 지수	당일 변화량 기준 단순 곱셈
shock_fatigue_rate	기계적 충격	$\frac{s191_t}{s9_t + 1}$	시간당 외부 충격 누적 피로도	역방향 뒤집어 Rate로 통일
s191_28d_sum	기계적 충격	$\sum_{i=0}^{27} \Delta E_{t-i}$	외부 충격 28일 총합	28일 내 장기 누적된 외부 물리 충격 횟수 합산
s191_28d_max	기계적 충격	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	외부 충격 28일 최댓값	28일 내 발생한 외부 물리적 충격의 최대치
s191_14d_sum	기계적 충격	$\sum_{i=0}^{13} \Delta E_{t-i}$	외부 충격 14일 총합	14일 내 누적된 외부 물리 충격 횟수 합산
s191_14d_max	기계적 충격	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	외부 충격 14일 최댓값	14일 내 발생한 외부 물리적 충격(G-Sense)의 최대치
s191_diff	기계적 충격	$\Delta s191_t$	G-Sense Error 변화량	충격 발생 빈도 증가
smart_191_raw	기계적 충격		G-Sense Error 원본 값	누적 충격 횟수
temp_error_index	열 스트레스	$\max(0, s194_t - 40) \times E_t$	고온 환경 에러 발생 지수	40도 이하 강제 0 처리 후 당일 에러 총합 곱셈
thermal_fatigue_integral_7d	열 스트레스	$\sum_{i=0}^{6} \max(0, s194_{t-i} - 40)$	7일간 40도 초과 열 피로 누적량(면적)	Physics-style 적분 연산 (유지)
s190_28d_ewma	열 스트레스	$0.069 \cdot T_t + 0.931 \cdot E_{t-1}$	기류 온도 28일 지수가중	장기 28일 내 온도 변화 추이의 지수가중 이동평균
s190_28d_dai	열 스트레스	$\frac{T_t - T_{t-27}}{27}$	기류 온도 28일 평균증가량	28일간 만성적인 쿨링 성능 저하 여부 추세
s190_28d_cid	열 스트레스	$\sqrt{\sum_{i=0}^{27}(T_{t-i} - T_{t-i-1})^2}$	기류 온도 28일 불변거리	28일 온도 시계열 장기 복잡도 및 누적 열충격
s190_28d_asfd	열 스트레스	$\sum_{i=0}^{27} \lvert T_{t-i} - T_{t-i-1} \rvert$	기류 온도 28일 차분절대합	28일 장기적 온도 변화로 누적된 열 피로도 총합
s190_28d_zscore	열 스트레스	$\frac{T_t - \mu_{28}}{\sigma_{28} + 1e-5}$	기류 온도 28일 Z스코어	장기 28일 평균 대비 당일 온도의 통계적 이상 수치
s190_28d_std	열 스트레스	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(T_{t-i} - \mu_{28})^2}$	기류 온도 28일 표준편차	28일간 기류 온도의 장기 불규칙 산포도
s190_28d_mean	열 스트레스	$\frac{1}{28}\sum_{i=0}^{27} T_{t-i}$	기류 온도 28일 평균	28일간 계절성 포함 평균 내부 기류 온도 상태
s190_28d_max	열 스트레스	$\max(T_{t-27}, \dots, T_t)$	기류 온도 28일 최댓값	28일간 기록된 내부 기류 온도의 최대치
s190_14d_ewma	열 스트레스	$0.133 \cdot T_t + 0.867 \cdot E_{t-1}$	기류 온도 14일 지수가중	최근 14일 내 기류 온도 변화의 지수가중 이동평균
s190_14d_dai	열 스트레스	$\frac{T_t - T_{t-13}}{13}$	기류 온도 14일 평균증가량	14일간 일평균 내부 기류 온도 상승/하락 선형 추세
s190_14d_cid	열 스트레스	$\sqrt{\sum_{i=0}^{13}(T_{t-i} - T_{t-i-1})^2}$	기류 온도 14일 불변거리	14일간 온도 시계열의 복잡도 및 열충격 스케일
s190_14d_asfd	열 스트레스	$\sum_{i=0}^{13} \lvert T_{t-i} - T_{t-i-1} \rvert$	기류 온도 14일 차분절대합	14일간 냉각/발열을 오가며 발생한 중기 열 변화 거리
s190_14d_zscore	열 스트레스	$\frac{T_t - \mu_{14}}{\sigma_{14} + 1e-5}$	기류 온도 14일 Z스코어	과거 14일 평균 대비 당일 온도의 통계적 이상 수치
s190_14d_std	열 스트레스	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(T_{t-i} - \mu_{14})^2}$	기류 온도 14일 표준편차	14일간 내부 기류 온도의 불규칙한 산포도
s190_14d_mean	열 스트레스	$\frac{1}{14}\sum_{i=0}^{13} T_{t-i}$	기류 온도 14일 평균	14일간 평균 내부 기류 온도 상태
s190_14d_max	열 스트레스	$\max(T_{t-13}, \dots, T_t)$	기류 온도 14일 최댓값	14일간 기록된 내부 기류 온도의 최대치
s190_7d_ewma	열 스트레스	$0.25 \cdot T_t + 0.75 \cdot EWMA_{t-1}$	기류 온도 7일 지수가중	최근 7일 내 급성 냉각 문제에 가중치를 둔 추세
s190_7d_dai	열 스트레스	$\frac{T_t - T_{t-6}}{6}$	기류 온도 7일 평균증가량	7일간 일평균 내부 기류 온도 상승/하락 선형 추세
s190_7d_cid	열 스트레스	$\sqrt{\sum_{i=0}^{6}(T_{t-i} - T_{t-i-1})^2}$	기류 온도 7일 불변거리	7일간 온도 시계열의 복잡도 및 열충격 스케일
s190_7d_asfd	열 스트레스	$\sum_{i=0}^{6} \lvert T_{t-i} - T_{t-i-1} \rvert$	기류 온도 7일 차분절대합	7일간 냉각/발열을 오가며 발생한 총 열 변화 거리
s190_7d_zscore	열 스트레스	$\frac{T_t - \mu_7}{\sigma_7 + 1e-5}$	기류 온도 7일 Z스코어	과거 7일 평균 대비 당일 온도의 통계적 이상 수치
s190_7d_std	열 스트레스	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(T_{t-i} - \mu_{7})^2}$	기류 온도 7일 표준편차	7일간 내부 기류 온도의 불규칙한 산포도
s190_7d_mean	열 스트레스	$\frac{1}{7}\sum_{i=0}^{6} T_{t-i}$	기류 온도 7일 평균	7일간 평균 내부 기류 온도 상태
s190_diff	열 스트레스	$\Delta s190_t$	Airflow 온도 변화량	온도 변동성
smart_190_raw	열 스트레스		Airflow 온도 원본 값	내부 공기 흐름 지표
s194_over40_7d_count	온도 수준	$\sum_{i=0}^{6} \mathbb{I}(s194_{t-i} > 40)$	7일 중 온도가 40도를 초과한 일수	조건 기반 카운트 (유지)
thermal_stress_index	온도 수준	$\max(0, s194_t - 40)^2 \times \Delta s241_t$	고온 스트레스 및 누적 쓰기량 결합 지수	40도 초과분에 가중치(제곱) 부여
s194_28d_ewma	온도 수준	$0.069 \cdot T_t + 0.931 \cdot E_{t-1}$	드라이브 온도 28일 지수가중	장기 28일 내 발열 트렌드의 지수가중 이동평균
s194_28d_dai	온도 수준	$\frac{T_t - T_{t-27}}{27}$	드라이브 온도 28일 증가량	28일간 일평균 장기 만성 온도 상승 선형 추세
s194_28d_cid	온도 수준	$\sqrt{\sum_{i=0}^{27}(T_{t-i} - T_{t-i-1})^2}$	드라이브 온도 28일 거리	28일간 드라이브 표면에 누적된 만성적 열 충격 스케일
s194_28d_asfd	온도 수준	$\sum_{i=0}^{27} \lvert T_{t-i} - T_{t-i-1} \rvert$	드라이브 온도 28일 차분합	28일간 드라이브에 누적된 만성 열 피로도 총합
s194_28d_zscore	온도 수준	$\frac{T_t - \mu_{28}}{\sigma_{28} + 1e-5}$	드라이브 온도 28일 Z스코어	과거 28일 평균 발열 대비 당일의 장기 과열 이상치
s194_28d_std	온도 수준	$\sqrt{\frac{1}{28}\sum_{i=0}^{27}(T_{t-i} - \mu_{28})^2}$	드라이브 온도 28일 편차	28일간 온도 산포에 따른 장기간의 부하 불규칙성
s194_28d_mean	온도 수준	$\frac{1}{28}\sum_{i=0}^{27} T_{t-i}$	드라이브 온도 28일 평균	28일간 만성적인 드라이브 평균 발열 상태
s194_28d_max	온도 수준	$\max(T_{t-27}, \dots, T_t)$	드라이브 온도 28일 최댓값	28일간 기록된 드라이브 표면 온도 장기 최대치
s194_14d_ewma	온도 수준	$0.133 \cdot T_t + 0.867 \cdot E_{t-1}$	드라이브 온도 14일 지수가중	14일 내 드라이브 온도 변화의 지수가중 이동평균
s194_14d_dai	온도 수준	$\frac{T_t - T_{t-13}}{13}$	드라이브 온도 14일 증가량	14일간 일평균 드라이브 온도 선형 증가 추세
s194_14d_cid	온도 수준	$\sqrt{\sum_{i=0}^{13}(T_{t-i} - T_{t-i-1})^2}$	드라이브 온도 14일 거리	14일간 드라이브 표면에 가해진 중기 열 충격
s194_14d_asfd	온도 수준	$\sum_{i=0}^{13} \lvert T_{t-i} - T_{t-i-1} \rvert$	드라이브 온도 14일 차분합	14일간 드라이브에 발생한 중기 열 변화량 누적
s194_14d_zscore	온도 수준	$\frac{T_t - \mu_{14}}{\sigma_{14} + 1e-5}$	드라이브 온도 14일 Z스코어	과거 14일 발열 대비 당일의 중기 과열 이상치
s194_14d_std	온도 수준	$\sqrt{\frac{1}{14}\sum_{i=0}^{13}(T_{t-i} - \mu_{14})^2}$	드라이브 온도 14일 편차	14일간 드라이브 온도의 중기 산포도
s194_14d_mean	온도 수준	$\frac{1}{14}\sum_{i=0}^{13} T_{t-i}$	드라이브 온도 14일 평균	14일간 드라이브 평균 발열 상태
s194_14d_max	온도 수준	$\max(T_{t-13}, \dots, T_t)$	드라이브 온도 14일 최댓값	14일간 기록된 드라이브 표면 온도의 중기 최대치
s194_7d_ewma	온도 수준	$0.25 \cdot T_t + 0.75 \cdot EWMA_{t-1}$	드라이브 온도 7일 지수가중	7일 내 드라이브 과열 상태에 가중치를 둔 추세
s194_7d_dai	온도 수준	$\frac{T_t - T_{t-6}}{6}$	드라이브 온도 7일 증가량	7일간 일평균 드라이브 온도 선형 증가 추세
s194_7d_cid	온도 수준	$\sqrt{\sum_{i=0}^{6}(T_{t-i} - T_{t-i-1})^2}$	드라이브 온도 7일 거리	7일간 드라이브 표면에 가해진 단기 열 충격
s194_7d_asfd	온도 수준	$\sum_{i=0}^{6} \lvert T_{t-i} - T_{t-i-1} \rvert$	드라이브 온도 7일 차분합	7일간 드라이브 수축/팽창 변화의 이동 거리
s194_7d_zscore	온도 수준	$\frac{T_t - \mu_7}{\sigma_7 + 1e-5}$	드라이브 온도 7일 Z스코어	과거 7일 발열 대비 당일의 비정상적 과열 감지
s194_7d_std	온도 수준	$\sqrt{\frac{1}{7}\sum_{i=0}^{6}(T_{t-i} - \mu_{7})^2}$	드라이브 온도 7일 편차	7일간 드라이브 온도의 불안정한 요동 수준
s194_7d_mean	온도 수준	$\frac{1}{7}\sum_{i=0}^{6} T_{t-i}$	드라이브 온도 7일 평균	7일간 드라이브 평균 발열 상태
s194_7d_max	온도 수준	$\max(T_{t-6}, \dots, T_t)$	드라이브 온도 7일 최댓값	7일간 기록된 드라이브 표면 온도의 단기 최대치
s194_diff	온도 수준	$\Delta s194_t$	HDA 온도 변화량	온도 상승 가속도
smart_194_raw	온도 수준		HDA 온도 원본 값	하드웨어 절대 온도
zero_to_hero_count	시스템성 실패	$\sum_{k \in \mathcal{K}} \mathbb{I}(s_{k,t} > 0 \land s_{k,t-1} = 0)$	0이었다가 갑자기 튄 지표의 개수	[집합 K 확정] 타겟 변수 명확화
s184_1d_crash_flag	시스템성 실패	$\mathbb{I}(\Delta s184_t > 0)$	치명적 패리티 오류 1일 발생 플래그	단순 발생 여부 확인
data_corruption_hazard	시스템성 실패	$\mathbb{I}(\Delta s184_t > 0 \lor \Delta s199_t > 0)$	논리적 데이터 오염 위험 경고 플래그	패리티 오류 또는 통신 오류 발생
s184_14d_sum	시스템성 실패	$\sum_{i=0}^{13} \Delta E_{t-i}$	전송 경로 오류 14일 총합	14일 내 발생한 전송 경로 오류 장기 누적 데미지
s184_14d_max	시스템성 실패	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	전송 경로 오류 14일 최댓값	14일 내 발생한 전송 경로 오류 최대 피크
s184_7d_sum	시스템성 실패	$\sum_{i=0}^{6} \Delta E_{t-i}$	전송 경로 오류 7일 총합	7일 내 발생한 전송 경로 오류 누적 데미지
s184_7d_max	시스템성 실패	$\max(\Delta E_{t-6}, \dots, \Delta E_t)$	전송 경로 오류 7일 최댓값	7일 내 발생한 전송 경로 오류 최대 피크
s184_3d_sum	시스템성 실패	$\sum_{i=0}^{2} \Delta E_{t-i}$	전송 경로 오류 3일 총합	3일 내 발생한 전송 경로 오류 급성 누적 데미지
s184_3d_max	시스템성 실패	$\max(\Delta E_{t-2}, \dots, \Delta E_t)$	전송 경로 오류 3일 최댓값	3일 내 발생한 패리티 오류(End-to-End)의 급성 피크
smart_184_raw	시스템성 실패		전송 경로 오류 원본 값	상태 레벨
s184_diff	시스템성 실패	$\Delta s184_t$	전송 경로 오류 1일 변화량	급성 오류 증가
firmware_struggle_index	펌웨어 실패	$(\Delta s197_t + \Delta s198_t) \times \Delta Timeout\_Total_t$	펌웨어 오류 복구 지연(발악) 지수	변화량 기준 단순 곱셈
s198_error_rate	Sector 열화	$\frac{s198_t}{s9_t + 1}$	시간당 복구 불가 섹터 발생률	Inverse 뒤집어 Rate로 통일
s187_error_rate	Sector 열화	$\frac{s187_t}{s9_t + 1}$	시간당 복구 불가 오류 발생률	Inverse 뒤집어 Rate로 통일
error_density_14d	Sector 열화	$\frac{\sum_{i=0}^{13} E_{t-i}}{\sum_{i=0}^{13} \max(0, \Delta s9_{t-i}) + 1}$	14일간 시간 대비 에러 밀도	[논리 위험 수정] 분모를 14일간의 실제 누적 구동 시간($\sum \Delta s9$)으로 변경
s197_recovery_flag	Sector 열화	$\mathbb{I}(s197_t < s197_{t-1})$	불안정 섹터 회복 이벤트 플래그	전일 대비 누적 수치 감소 여부
error_saturation_score	Sector 열화	$\mathbb{I}(s187_t \ge 65535) + \mathbb{I}(s189_t \ge 65535) + \mathbb{I}(Seek\_Error_t \ge 65535)$	핵심 오류 지표 포화도 점수	임계치 도달 여부 확인
cumulative_error_score	Sector 열화	$5\lvert\Delta s5_t\rvert + 4\lvert\Delta s187_t\rvert + 3\lvert\Delta Timeout\_Total_t\rvert + 2\lvert\Delta s197_t\rvert + 1\lvert\Delta s198_t\rvert$	핵심 5대 변수 종합 위험 점수	에러 복구(음수)도 이벤트로 간주하여 절댓값 가중 합산
s198_28d_sum	Sector 열화	$\sum_{i=0}^{27} \Delta E_{t-i}$	복구 불가 섹터 28일 총합	28일 내 물리적으로 파괴된 섹터 장기 누적량
s198_28d_max	Sector 열화	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	복구 불가 섹터 28일 최댓값	28일 내 발생한 오프라인 복구 불가 섹터의 최대 피크
s198_14d_sum	Sector 열화	$\sum_{i=0}^{13} \Delta E_{t-i}$	복구 불가 섹터 14일 총합	14일 내 물리적으로 파괴된 섹터의 총 누적량
s198_14d_max	Sector 열화	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	복구 불가 섹터 14일 최댓값	14일 내 발생한 오프라인 복구 불가 섹터의 최대 피크
s197_28d_sum	Sector 열화	$\sum_{i=0}^{27} \Delta E_{t-i}$	불안정 섹터 28일 총합	28일 내 대기 섹터로 전환된 장기 누적 횟수
s197_28d_max	Sector 열화	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	불안정 섹터 28일 최댓값	28일 내 대기 섹터 증가량의 최대 피크
s197_14d_sum	Sector 열화	$\sum_{i=0}^{13} \Delta E_{t-i}$	불안정 섹터 14일 총합	14일 내 대기 섹터로 전환된 총 누적 횟수
s197_14d_max	Sector 열화	$\max(\Delta E_{t-13}, \dots, \Delta E_t)$	불안정 섹터 14일 최댓값	14일 내 대기 섹터(Pending) 증가량의 최대 피크
s187_28d_sum	Sector 열화	$\sum_{i=0}^{27} \Delta E_{t-i}$	복구 불가 오류 28일 총합	28일 내 발생한 치명적 복구 불가 장기 누적량
s187_28d_max	Sector 열화	$\max(\Delta E_{t-27}, \dots, \Delta E_t)$	복구 불가 오류 28일 최댓값	28일 내 발생한 복구 불가 에러 최대 피크
smart_187_raw	Sector 열화		복구 불가 오류 원본 값	상태 레벨
s187_diff	Sector 열화	$\Delta s187_t$	복구 불가 오류 1일 변화량	burst 탐지 핵심
smart_197_raw	Sector 열화		pending sector 원본 값	상태 레벨
s197_diff	Sector 열화	$\Delta s197_t$	pending sector 변화량	회복/악화 신호
smart_198_raw	Sector 열화		reallocated sector 원본 값	물리 손상 누적
s198_diff	Sector 열화	$\Delta s198_t$	reallocated sector 변화량	치명도 상승 신호
reallocated_pending_ratio	Reallocated / Pending	$\frac{s197_t + 1}{s5_t + 1}$	불량 섹터 대비 불안정 섹터 비율	누적 원본값 기준 비율 연산
pending_to_offline_ratio	Reallocated / Pending	$\frac{s197_t + 1}{s198_t + 1}$	대기 섹터의 복구 불가 전이 심각도	누적 원본값 기준 비율 연산
late_stage_degradation	지속 악화	$\max(0, \Delta s5_t) \times s9_t$	노후화 말기 불량 섹터 발생 위험도	불량 섹터 감소분은 0 처리 후 누적 시간 곱셈
error_growth_ratio	지속 악화	$\frac{E_t + 1}{E_{t-1} + 1}$	오류 가속화 비율	전일 대비 당일 에러 발생 비율
s197_7d_straight_rise	지속 악화	$\mathbb{I}\left( \sum_{i=0}^{6} \mathbb{I}(\Delta s197_{t-i} > 0) \ge 5 \right)$	7일 중 5일 이상 연속 상승 여부	이중 지시 함수를 통한 논리 연산 (유지)
timeout_total_days_since_last	최근 발생 시점	$\begin{cases} t - \max\{\tau \mid \max(0, \Delta Timeout\_Total_\tau) > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	마지막 Timeout 증가 이후 경과일	[표준 cases] 순수 증가 이벤트 기준
s187_days_since_first	최초 발생 시점	$\begin{cases} t - \min\{\tau \mid s187_\tau > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	smart_187 최초 발생 후 경과일	[표준 cases] 미발생 시 -1 반환
s5_days_since_first	최초 발생 시점	$\begin{cases} t - \min\{\tau \mid s5_\tau > 0\}, & \text{if } \exists \tau \\ -1, & \text{otherwise} \end{cases}$	smart_5 최초 발생 후 경과일	[표준 cases] 미발생 시 -1 반환
s5_ever_flag	과거 누적 발생	$\mathbb{I}(\max_{0 \le i \le t}(s5_i) > 0)$	불량 섹터 누적 발생 플래그	과거부터 현재까지 발생 이력 확인
s187_ever_flag	과거 누적 발생	$\mathbb{I}(\max_{0 \le i \le t}(s187_i) > 0)$	복구 불가 오류 횟수 누적 발생 플래그	과거부터 현재까지 발생 이력 확인
timeout_5s_damaged	직접 손상 발생	$\mathbb{I}(Timeout\_5s_t > 0)$	5초 이상 지연 손상 여부	누적 원본값 기준 발생 확인
seek_damaged	직접 손상 발생	$\mathbb{I}(Seek\_Error\_Count_t > 0)$	탐색 오류 손상 여부	누적 원본값 기준 발생 확인
s5_damaged	직접 손상 발생	$\mathbb{I}(s5_t > 0)$	불량 섹터 손상 여부	누적 원본값 기준 발생 확인
s198_damaged	직접 손상 발생	$\mathbb{I}(s198_t > 0)$	복구 불가 섹터 손상 여부	누적 원본값 기준 발생 확인
s197_damaged	직접 손상 발생	$\mathbb{I}(s197_t > 0)$	불안정 섹터 손상 여부	누적 원본값 기준 발생 확인
s187_damaged	직접 손상 발생	$\mathbb{I}(s187_t > 0)$	복구 불가 오류 손상 여부	누적 원본값 기준 발생 확인
smart_5_raw	직접 손상 발생		불량 섹터 원본 값	상태 레벨
s5_diff	직접 손상 발생	$\Delta s5_t$	불량 섹터 1일 변화량	악화 속도
s190_7d_max	온도 수준			
total_seeks_diff	총 탐색량			
total_seeks_7d_accel	총 탐색량			
total_seeks_28d_accel	총 탐색량			
total_seeks_14d_accel	총 탐색량			
total_seeks	총 탐색량			
total_reads_diff	총 읽기량			
total_reads_7d_accel	총 읽기량			
total_reads_28d_accel	총 읽기량			
total_reads_14d_accel	총 읽기량			
total_reads	총 읽기량			
seek_error_count	총 읽기량			
seek_error_14d_spike_ratio	Seek 경로 이상			
s5_28d_sum	Seek 경로 이상			
s5_28d_max	Reallocated / Pending			
s5_14d_sum	Reallocated / Pending			
s5_14d_max	Reallocated / Pending			
s242_28d_accel	Reallocated / Pending			
s242_14d_accel	총 읽기량			
s241_7d_accel	총 읽기량			
s241_28d_accel	총 기록량			
s241_14d_ewma	총 기록량			
s241_14d_dai	총 기록량			
s241_14d_accel	총 기록량			