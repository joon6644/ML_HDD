# --- notebooks\09a_global_interpretation.ipynb ---
import sys, os, json, joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.abspath(".."))

%load_ext autoreload
%autoreload 2

import config.interp_config as cfg
import config.train_config as tcfg
import config.final_eval_config as fe_cfg
from src.eval_core import prepare_disk_level_data, get_detailed_disk_records

warnings.filterwarnings("ignore")

# 한글 폰트 설정
for _font in ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
    if any(_font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False

print("✅ 환경 준비 완료")


import shap

# SHAP 계산용 5:5 균형 추출 (고장 패턴을 전역 해석에서 뚜렷하게 보기 위함)
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
    try:
        import subprocess
        subprocess.run(['git', 'add', str(shap_cache_path)], check=True)
        print(f'\n✅  SHAP 계산 및 캐시 저장(Git Add 자동화) 완료!')
    except Exception as e:
        print(f'\n✅  SHAP 계산 및 캐시 저장 완료! (Git 자동 등록 실패: {e})')


# §9.1 Summary Plot (Beeswarm)
print("[§9.1] SHAP Summary Plot (전역 해석)")
shap.summary_plot(
    mean_shap,
    X_sample,
    feature_names=FEATURE_COLS,
    max_display=cfg.SHAP_MAX_DISPLAY,
    show=True,
    plot_size=(10, 8),
)

# §9.1 Bar Plot — 절댓값 평균 기준 피처 중요도
print("[§9.1] SHAP Bar Plot (Mean |SHAP| 기준 피처 중요도)")
shap.summary_plot(
    mean_shap,
    X_sample,
    feature_names=FEATURE_COLS,
    max_display=cfg.SHAP_MAX_DISPLAY,
    plot_type="bar",
    show=True,
    plot_size=(10, 8),
)

# 중요도 순위 테이블 출력
mean_abs_shap = np.abs(mean_shap).mean(axis=0)
shap_rank_df = (
    pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
shap_rank_df.index += 1
print("\n📊 피처 중요도 순위 (Mean |SHAP|):")
print(shap_rank_df.to_markdown())

# --- notebooks\09b_true_positive_analysis.ipynb ---
import sys, os, json, joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.abspath('..'))

%load_ext autoreload
%autoreload 2

import config.train_config as tcfg
import config.final_eval_config as fe_cfg

warnings.filterwarnings('ignore')

for _font in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    if any(_font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

print('✅ 환경 준비 완료')


# --- notebooks\09c_normal_false_alarm_analysis.ipynb ---
import sys, os, json, joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.abspath('..'))

%load_ext autoreload
%autoreload 2

import config.train_config as tcfg
import config.final_eval_config as fe_cfg

warnings.filterwarnings('ignore')

for _font in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    if any(_font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

print('✅ 환경 준비 완료')


# --- notebooks\09d_early_alarm_analysis.ipynb ---
import sys, os, json, joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.abspath('..'))

%load_ext autoreload
%autoreload 2

import config.train_config as tcfg
import config.final_eval_config as fe_cfg

warnings.filterwarnings('ignore')

for _font in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    if any(_font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

print('✅ 환경 준비 완료')


# --- notebooks\09e_false_negative_analysis.ipynb ---
import sys, os, json, joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.abspath('..'))

%load_ext autoreload
%autoreload 2

import config.train_config as tcfg
import config.final_eval_config as fe_cfg

warnings.filterwarnings('ignore')

for _font in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    if any(_font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

print('✅ 환경 준비 완료')


