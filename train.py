import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 讀取資料
train = pd.read_csv('happiness_train_2015_2018.csv')
test = pd.read_csv('pridict_2019.csv')

# 特徵欄位
features = [
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom to make life choices',
    'Generosity',
    'Perceptions of corruption',
]


train['Support_x_Freedom'] = train['Social support'] * train['Freedom to make life choices']
test['Support_x_Freedom'] = test['Social support'] * test['Freedom to make life choices']
train['GDP_x_Life'] = train['GDP per capita'] * train['Healthy life expectancy']
test['GDP_x_Life'] = test['GDP per capita'] * test['Healthy life expectancy']
train['GDP_x_Support'] = train['GDP per capita'] * train['Social support']
test['GDP_x_Support'] = test['GDP per capita'] * test['Social support']
train['LifeExp_x_Support'] = train['Healthy life expectancy'] * train['Social support']
test['LifeExp_x_Support'] = test['Healthy life expectancy'] * test['Social support']

# 最終特徵列表
features += ['Support_x_Freedom', 'GDP_x_Life','GDP_x_Support', 'LifeExp_x_Support']

# 特徵與目標
X_train = train[features].copy()
y_train = train['Score']

test['Year'] = 2019
X_test = test[features].copy()
y_test = test['Score']

# 建立與訓練模型（調整參數）
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# MSE 評估
mse = mean_squared_error(y_test, y_pred)
print(f"2019 MSE Loss: {mse:.4f}")
# R² 評估
r2 = r2_score(y_test, y_pred)
print(f"2019 R² Score: {r2:.4f}")

# 印出誤差最大前10名國家
result = test[['Country or region', 'Score']].copy()
result['Predicted'] = y_pred
result['AbsError'] = (result['Score'] - result['Predicted']).abs()
print(result.sort_values('AbsError', ascending=False).head(10))
# 新增排名欄位（數字越小，幸福度越高）
result['ActualRank'] = result['Score'].rank(ascending=False, method='min').astype(int)
result['PredictedRank'] = result['Predicted'].rank(ascending=False, method='min').astype(int)
# 計算排名誤差
result['RankDiff'] = (result['ActualRank'] - result['PredictedRank']).abs()
mean_rank_diff = result['RankDiff'].mean()
print(f"平均排名誤差: {mean_rank_diff:.2f}")
# 定義區間分類函數
def classify_group(rank):
    if rank <= 40:
        return 'Top'
    elif rank <= 80:
        return 'Upper-Mid'
    elif rank <= 120:
        return 'Lower-Mid'
    else:
        return 'Low'

# 應用分類
result['ActualGroup'] = result['ActualRank'].apply(classify_group)
result['PredictedGroup'] = result['PredictedRank'].apply(classify_group)

# 計算準確率
correct_predictions = (result['ActualGroup'] == result['PredictedGroup']).sum()
total_predictions = len(result)
accuracy = correct_predictions / total_predictions * 100

print(f"分區預測準確率: {accuracy:.2f}%")

# 存檔
result.to_csv('result.csv', index=False)
print("Result saved to result.csv")
