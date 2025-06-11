import pandas as pd
from scipy.stats import kstest, norm
import matplotlib.pyplot as plt

# 讀取整合資料
df = pd.read_csv('happiness_train_2015_2018.csv')

# 共同 feature 欄位
columns_to_test = [
    "Score", "GDP per capita", "Social support", 
    "Healthy life expectancy", "Freedom to make life choices", 
    "Generosity", "Perceptions of corruption"
]

years = [2015, 2016, 2017, 2018]

for year in df["Year"].unique():
    print(f"\n📅 Year: {year}")
    data_year = df[df["Year"] == year]
    for col in columns_to_test:
        values = data_year[col].dropna()
        # 標準化資料來對比標準常態分布
        standardized = (values - values.mean()) / values.std()
        stat, p = kstest(standardized, "norm")
        print(f"{col}: KS statistic = {stat:.4f}, p-value = {p:.4f}")