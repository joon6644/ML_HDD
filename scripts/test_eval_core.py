import numpy as np
import sys
sys.path.insert(0, r'C:\Workspace\06_ML_projdect\26_1_COIN')
from src.eval_core import ThresholdTuner, evaluate_row_level, EntityLevelEvaluator
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless

rng = np.random.default_rng(42)

# ── 합성 행 단위 데이터 ───────────────────────────────────────
n_pos, n_neg = 500, 50000
y_true = np.array([1]*n_pos + [0]*n_neg)
prob_pos = np.clip(rng.beta(5, 2, n_pos), 0, 1)
prob_neg = np.clip(rng.beta(1, 8, n_neg), 0, 1)
y_prob   = np.concatenate([prob_pos, prob_neg])

# --- §7 ThresholdTuner ---
print("=== §7 ThresholdTuner ===")
tuner = ThresholdTuner(max_fpr=0.01, n_grid=1000)
res   = tuner.fit(y_true, y_prob)
thr   = tuner.best_threshold
assert res['fpr'] <= 0.011, f"FPR 제약 위반: {res['fpr']}"
assert 0 <= res['recall'] <= 1
assert 0 <= res['pr_auc'] <= 1
print(f"threshold={thr:.4f}  recall={res['recall']:.4f}  fpr={res['fpr']:.4f}")
print("ThresholdTuner PASS")

# --- §8.1 evaluate_row_level ---
print("\n=== §8.1 evaluate_row_level ===")
r = evaluate_row_level(y_true, y_prob, thr, title='Synthetic')
assert 0 <= r['pr_auc'] <= 1
assert r['n_pos'] == n_pos
assert r['n_neg'] == n_neg
assert r['tp'] + r['fn'] == n_pos, f"tp+fn={r['tp']+r['fn']} != n_pos={n_pos}"
assert r['fp'] + r['tn'] == n_neg, f"fp+tn={r['fp']+r['tn']} != n_neg={n_neg}"
print("evaluate_row_level PASS")

# --- §8.2 EntityLevelEvaluator ---
print("\n=== §8.2 EntityLevelEvaluator ===")
entities = []
base_date = pd.Timestamp('2022-01-01')
for i in range(100):
    sn = f'FAIL_{i:03d}'
    for d in range(30):
        entities.append({'serial_number': sn,
                         'date': base_date + pd.Timedelta(days=d),
                         'failure': int(d >= 20)})
for i in range(200):
    sn = f'NORM_{i:03d}'
    for d in range(30):
        entities.append({'serial_number': sn,
                         'date': base_date + pd.Timedelta(days=d),
                         'failure': 0})
df_ent = pd.DataFrame(entities)

rng2   = np.random.default_rng(0)
probs2 = []
for _, row in df_ent.iterrows():
    if row['failure'] == 1:
        probs2.append(float(rng2.beta(6, 1)))
    elif str(row['serial_number']).startswith('FAIL'):
        probs2.append(float(rng2.beta(4, 3)))
    else:
        probs2.append(float(rng2.beta(1, 9)))
y_prob_ent = np.array(probs2)

ev     = EntityLevelEvaluator()
result = ev.evaluate(df_ent, y_prob_ent, threshold=0.5)

assert result['n_failure_entities'] == 100
assert result['n_normal_entities']  == 200
assert result['hit_count'] + result['miss_count'] == 100
assert abs(result['hit_rate'] + result['miss_rate'] - 1.0) < 1e-9
assert 0.0 <= result['fa_rate'] <= 1.0

# Lead time 검증: 탐지된 고장 개체의 lead time 은 항상 0 이상이어야 함
ed = result['entity_df']
hit_lt = ed[(ed['is_failure']==1) & (ed['has_alarm']==1)]['lead_time_days']
assert (hit_lt >= 0).all(), f"음수 lead time 존재: {hit_lt.min()}"

print(f"hit_rate={result['hit_rate']:.3f}  miss_rate={result['miss_rate']:.3f}  fa_rate={result['fa_rate']:.3f}")
print(f"mean_lead_time={result['mean_lead_time_days']:.1f}일")
print("EntityLevelEvaluator PASS")

print("\n>>> 모든 테스트 통과!")
