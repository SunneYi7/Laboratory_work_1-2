import pandas as pd

# 读取原始数据，确保所有列都以数值形式加载
weather_data = pd.read_csv("data/Summary_of_Weather.csv", low_memory=False)
stations_data = pd.read_csv("data/Weather_Station_Locations.csv", low_memory=False)

# 🚀 强制转换所有列为数值型，非数值内容会变成 NaN
weather_data = weather_data.apply(pd.to_numeric, errors='coerce')

# 选取特征列
features = weather_data[['Precip', 'WindGustSpd', 'Snowfall']]

# 🚀 处理 NaN（填充均值）
features.fillna(features.mean(), inplace=True)

# 🚀 再次检查 NaN 是否全部去除
if features.isnull().sum().sum() > 0:
    print("❌ 仍然有 NaN，数据预处理失败！")
    exit()

# 保存预处理数据
features.to_csv("processed_data.csv", index=False)
print("✅ 数据预处理完成，文件已保存为 processed_data.csv")
