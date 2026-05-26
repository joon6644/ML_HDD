import json

nb_file = 'notebooks/09f_failure_observability_analysis.ipynb'
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new source code for cell 5
new_source = """# 각 디스크별 수명 말기 (D-30 ~ D-0) 최대값 추출
# D-n 계산을 위해 디스크별 레코드 수집
last_30d_stats = []

for serial in tp_serials + hard_miss_serials:
    df_s = df_failed[df_failed[cfg.SERIAL_COL] == serial]
    # 마지막 30일만 추출 (데이터는 이미 오름차순 가정)
    df_s_tail = df_s.tail(30)
    
    group = 'TP' if serial in tp_serials else 'FN_Hard'
    
    # 핵심 피처의 이상 징후 확인
    max_err_density = df_s_tail['error_density_14d'].max()
    max_198_raw = df_s_tail['smart_198_raw'].max()
    max_187_raw = df_s_tail['smart_187_raw'].max() if 'smart_187_raw' in df_s_tail.columns else 0
    max_5_raw = df_s_tail['smart_5_raw'].max() if 'smart_5_raw' in df_s_tail.columns else 0
    
    # 어떤 형태의 에러라도 축적되었는가?
    has_anomaly = (max_err_density > 0) or (max_198_raw > 0) or (max_187_raw > 0) or (max_5_raw > 0)
    
    last_30d_stats.append({
        'serial': serial,
        'group': group,
        'max_err_density': max_err_density,
        'max_198_raw': max_198_raw,
        'has_anomaly': has_anomaly
    })

df_stats = pd.DataFrame(last_30d_stats)

# 그룹별 이상 징후 발현 비율
silent_ratio = df_stats.groupby('group')['has_anomaly'].value_counts(normalize=True).unstack().fillna(0) * 100

print("📊 수명 말기(마지막 30일) 동안 핵심 에러(Density, 198, 187 등)가 한 번이라도 관측된 비율")
print(silent_ratio)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#ffffff')
silent_ratio.loc[['TP', 'FN_Hard']].plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#3498db'])

# Add annotations to the bars
for i, group in enumerate(['TP', 'FN_Hard']):
    # The observable percentage is the True value (index 1 if False is index 0)
    observable_pct = silent_ratio.loc[group, True]
    silent_pct = silent_ratio.loc[group, False]
    
    # Text for Observable
    if observable_pct > 0:
        ax.text(i, silent_pct + observable_pct/2, f"{observable_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    # Text for Silent
    if silent_pct > 0:
        ax.text(i, silent_pct/2, f"{silent_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)

ax.set_title("Observable Precursor Presence in TP vs Hard FN Failures\\n(within the final 30 days before failure)", fontsize=15, fontweight='bold')
ax.set_ylabel("Percentage of Disks (%)", fontsize=12)
ax.set_xlabel("Failure Category", fontsize=12)
ax.legend(['Silent (No Anomaly)', 'Observable (Has Anomaly)'], title='Precursor Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""

nb['cells'][5]['source'] = [line + '\n' for line in new_source.split('\n')]
# Remove the last empty newline added
nb['cells'][5]['source'][-1] = nb['cells'][5]['source'][-1].strip('\n')

with open(nb_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Modified cell 5 plotting logic successfully!")
