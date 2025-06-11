import pandas as pd

# 欄位對應表，標準化為2019的欄位名稱
columns_map = {
    'Country or region': ['Country', 'Country or region'],
    'Score': ['Happiness Score', 'Happiness.Score', 'Score'],
    'GDP per capita': ['Economy (GDP per Capita)', 'Economy..GDP.per.Capita.', 'GDP per capita'],
    'Social support': ['Family', 'Social support'],
    'Healthy life expectancy': ['Health (Life Expectancy)', 'Health..Life.Expectancy.', 'Healthy life expectancy'],
    'Freedom to make life choices': ['Freedom', 'Freedom to make life choices'],
    'Generosity': ['Generosity'],
    'Perceptions of corruption': ['Trust (Government Corruption)', 'Trust..Government.Corruption.', 'Perceptions of corruption']
}

def standardize_columns(df, columns_map):
    col_rename = {}
    for std_col, possible_cols in columns_map.items():
        for col in possible_cols:
            if col in df.columns:
                col_rename[col] = std_col
                break
    # 只保留需要的欄位
    return df.rename(columns=col_rename)[list(columns_map.keys())]

# 讀取並標準化，並加上年份欄位
df_2015 = standardize_columns(pd.read_csv('2015.csv'), columns_map)
df_2015['Year'] = 2015
df_2016 = standardize_columns(pd.read_csv('2016.csv'), columns_map)
df_2016['Year'] = 2016
df_2017 = standardize_columns(pd.read_csv('2017.csv'), columns_map)
df_2017['Year'] = 2017
df_2018 = standardize_columns(pd.read_csv('2018.csv'), columns_map)
df_2018['Year'] = 2018

# 合併
df_all = pd.concat([df_2015, df_2016, df_2017, df_2018], ignore_index=True)
df_all.to_csv('happiness_train_2015_2018.csv', index=False)