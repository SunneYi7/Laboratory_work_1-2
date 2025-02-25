import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

# 读取处理后的数据
data = pd.read_csv("processed_data.csv")

# 目标变量
target = pd.read_csv("data/Summary_of_Weather.csv")["MeanTemp"]

# 读取训练好的模型
model = joblib.load("linear_regression_model.pkl")

# 进行预测
y_pred = model.predict(data)

# 计算模型指标
mse = mean_squared_error(target, y_pred)
r2 = r2_score(target, y_pred)

# 保存评估结果
metrics = {"Mean Squared Error": mse, "R^2 Score": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ 模型评估完成，指标已存入 metrics.json")
