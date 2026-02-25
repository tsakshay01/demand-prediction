import requests
import json

r = requests.post('http://localhost:5000/evaluate', json={
    'file_path': 'c:/Users/tsaks/OneDrive/Desktop/proj_demand/public/realistic_test_data.csv'
})
d = r.json()
s = d['summary']

print('')
print('=' * 55)
print('  DEMAND PREDICTION - EVALUATION REPORT')
print('=' * 55)
print(f'  Products Evaluated  : {s["total_products_evaluated"]}')
print(f'  Total Datapoints    : {s["total_datapoints"]}')
print('-' * 55)
print(f'  MAE  (Mean Abs Error)   : {s["mae"]} units')
print(f'  RMSE (Root Mean Sq Err) : {s["rmse"]} units')
print(f'  MAPE (Mean Abs % Error) : {s["mape"]}%')
print(f'  Accuracy (100 - MAPE)   : {s["accuracy_pct"]}%')
print(f'  R2 Score                : {s["r2_score"]}')
print(f'  Direction Accuracy      : {s["direction_accuracy"]}%')
print('=' * 55)
print('')
print(f'{"Product":<40} {"MAE":>7} {"MAPE%":>7} {"Acc%":>7} {"R2":>10} {"Dir%":>7}')
print('-' * 78)
for p in d['per_product']:
    print(f'  {p["description"][:38]:<38} {p["mae"]:>7} {p["mape"]:>7} {p["accuracy_pct"]:>7} {p["r2_score"]:>10} {p["direction_accuracy"]:>7}')
print('')
