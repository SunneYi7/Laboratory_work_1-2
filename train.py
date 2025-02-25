import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# 读取处理后的数据
data = pd.read_csv("processed_data.csv")

# 目标变量
target = pd.read_csv("data/Summary_of_Weather.csv")["MeanTemp"]

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "linear_regression_model.pkl")

print("✅ 线性回归模型训练完成，已保存为 linear_regression_model.pkl")
