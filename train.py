import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer  # 导入 SimpleImputer
import joblib

# 读取处理后的数据
data = pd.read_csv("processed_data.csv")

# 目标变量
target = pd.read_csv("data/Summary_of_Weather.csv")["MeanTemp"]

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 🚀 **1. 使用 SimpleImputer 填充 NaN**
imputer = SimpleImputer(strategy='mean')  # 使用均值填充 NaN
X_train = imputer.fit_transform(X_train)  # 填充训练数据
X_test = imputer.transform(X_test)  # 填充测试数据

# 处理目标变量中的 NaN
y_train = y_train.fillna(y_train.mean())
y_test = y_test.fillna(y_test.mean())

# 🚀 **2. 确保没有 NaN**
if pd.isnull(X_train).sum().sum() > 0 or pd.isnull(y_train).sum() > 0:
    print("❌ 训练数据中仍然存在 NaN，请检查！")
    exit()

# 🚀 **3. 训练线性回归模型**
model = LinearRegression()
model.fit(X_train, y_train)

# 🚀 **4. 保存模型**
joblib.dump(model, "linear_regression_model.pkl")

print("✅ 线性回归模型训练完成，已保存为 linear_regression_model.pkl")
