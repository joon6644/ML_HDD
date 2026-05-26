import json

with open('notebooks/08a_add_failure_date.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        source = source.replace('08a. Add Failure Date Rows to Test Dataset', '07d. Add Failure Date Rows to Val_Calib Dataset')
        source = source.replace('test.parquet', 'val_calib.parquet')
        source = source.replace('test_with_failure_date.parquet', 'val_calib_with_failure_date.parquet')
        source = source.replace('Test 데이터', 'Val_Calib 데이터')
        source = source.replace('기존 Test 데이터', '기존 Val_Calib 데이터')
        # write back
        import io
        cell['source'] = [l for l in io.StringIO(source)]
    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        source = source.replace('test.parquet', 'val_calib.parquet')
        source = source.replace('test_with_failure_date.parquet', 'val_calib_with_failure_date.parquet')
        source = source.replace('TEST_PATH', 'VAL_CALIB_PATH')
        source = source.replace('df_test', 'df_val_calib')
        source = source.replace('df_fail_test', 'df_fail_val_calib')
        source = source.replace('failed_serials_test', 'failed_serials_val_calib')
        source = source.replace('serial_number_test', 'serial_number_val_calib')
        source = source.replace('sn_test', 'sn_val_calib')
        source = source.replace('test dataset rows', 'val_calib dataset rows')
        # write back
        import io
        cell['source'] = [l for l in io.StringIO(source)]

with open('notebooks/07d_add_failure_date_calib.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
