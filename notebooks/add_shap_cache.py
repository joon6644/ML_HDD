import json

nb_path = r'c:\Workspace\06_ML_projdect\26_1_COIN\notebooks\09_model_interpretation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_shap = r"""# SHAP 계산용 5:5 균형 추출 (고장 패턴을 전역 해석에서 뚜렷하게 보기 위함)
sample_size = min(cfg.SHAP_SAMPLE_SIZE, len(df_test))
target_fail_count = sample_size // 2

# 1. 고장/정상 인덱스 분리
fail_indices = df_test[df_test[cfg.TARGET_COL] == 1].index
norm_indices = df_test[df_test[cfg.TARGET_COL] == 0].index

# 2. 고장 데이터 샘플링 (전체 샘플의 절반을 목표로 하되, 모자라면 전수 사용)
n_fail_sample = min(len(fail_indices), target_fail_count)
# 나머지를 정상 데이터로 채움
n_norm_sample = sample_size - n_fail_sample

fail_sample = df_test.loc[fail_indices].sample(n_fail_sample, random_state=cfg.SHAP_SEED)
norm_sample = df_test.loc[norm_indices].sample(n_norm_sample, random_state=cfg.SHAP_SEED)

# 3. 데이터셋 결합 및 셔플 (순서가 뭉치지 않게)
df_sample = pd.concat([fail_sample, norm_sample]).sample(frac=1, random_state=cfg.SHAP_SEED).reset_index(drop=True)

X_sample = df_sample[FEATURE_COLS]
y_sample = df_sample[cfg.TARGET_COL]

print(f"SHAP 계산 준비... (샘플={sample_size:,}, 모델={len(models)}개)")
print(f"  → 균형 튜닝 추출 완료: 고장 {n_fail_sample:,}개 / 정상 {n_norm_sample:,}개")

shap_cache_path = SAVE_DIR / f"shap_cache_seed{cfg.SHAP_SEED}_size{sample_size}.npy"

if os.path.exists(shap_cache_path):
    print(f"♻️ 전역 SHAP 캐시 로드 중: {shap_cache_path.name}")
    mean_shap = np.load(shap_cache_path)
    print(f"✅ SHAP 캐시 로드 완료!")
else:
    print("  ※ 모델 수 × 샘플 수에 비례하여 수 분 소요될 수 있습니다.")
    sv_list = []
    for i, model in enumerate(models):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]  # 이진 분류: 양성 클래스(고장) SHAP
        sv_list.append(sv)
        print(f"  모델 {i+1}/{len(models)} 완료", end="\r")
    
    mean_shap = np.mean(sv_list, axis=0)
    np.save(shap_cache_path, mean_shap)
    print(f"\n✅  SHAP 계산 및 캐시 저장 완료!")
"""

for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if '# SHAP 계산용 5:5 균형 추출' in s:
            c['source'] = [line + '\n' for line in cell_shap.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Success')
