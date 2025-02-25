import pandas as pd

# 读取原始数据，确保所有列都以数值形式加载
weather_data = pd.read_csv("data/Summary_of_Weather.csv", low_memory=False)

# 🚀 强制转换所有列为数值型，非数值内容会变成 NaN
weather_data = weather_data.apply(pd.to_numeric, errors='coerce')

# 打印一些列的统计信息，检查是否有问题
print(weather_data[['Precip', 'Snowfall', 'MaxTemp', 'MinTemp', 'PRCP', 'MO', 'YR', 'DA']].describe())

# 选取特征列（添加更多特征）
features = weather_data[['Precip', 'Snowfall', 'MaxTemp', 'MinTemp', 'PRCP', 'MO', 'YR', 'DA']]
target = weather_data['MeanTemp']

# 🚀 处理缺失值
print("\n🔍 缺失值统计（处理前）:")
print(features.isnull().sum())

# - 先用均值填充 NaN
# 为避免警告，使用 .loc 确保修改 DataFrame
features.loc[:, :] = features.fillna(features.mean())

# 处理目标变量
target = target.fillna(target.mean())

# - 再次检查 NaN 是否全部去除
print("\n✅ 缺失值统计（处理后）:")
print(features.isnull().sum())

# 🚀 **5. 确保数据没有 NaN**
if features.isnull().sum().sum() > 0 or target.isnull().sum() > 0:
    print("❌ 仍然有 NaN，数据预处理失败！")
    exit()

# 保存预处理数据
features.to_csv("processed_data.csv", index=False)
print("✅ 数据预处理完成，文件已保存为 processed_data.csv")
