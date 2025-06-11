import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv('happiness_train_2015_2018.csv')

# 欲分析的欄位
features = [
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom to make life choices',
    'Generosity',
    'Perceptions of corruption',
    'Score'
]

# 依年份繪製相關係數矩陣
for year in sorted(df['Year'].unique()):
    plt.figure(figsize=(8,6))
    corr = df[df['Year'] == year][features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for Year {year}')
    plt.tight_layout()
    plt.savefig(f'correlation_{year}.png')
    plt.show()
# 全年份數據的相關係數矩陣
plt.figure(figsize=(8, 6))
corr_all = df[features].corr()
print(corr_all.to_string())
sns.heatmap(corr_all, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for All Years')
plt.tight_layout()
plt.savefig('correlation_all_years.png')
plt.show()