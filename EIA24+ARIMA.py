import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error



# 数据归一化函数
def normalize_data(data):
    if isinstance(data, pd.Series):
        data=data.values

    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

# 计算RMSE
def rmse_loss(output, target):
    return np.sqrt(mean_squared_error(target, output))

# 计算MAE
def mae_loss(output, target):
    return mean_absolute_error(target, output)

# 计算MAPE
def mape_loss(output, target):
    return np.mean(np.abs((output - target) / target)) * 100

# 模型评估函数
def evaluate_model(train_series, test_series):
    model = auto_arima(train_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    model_fit = model.fit(train_series)
    forecast = model_fit.predict(n_periods=len(test_series))
    rmse = rmse_loss(forecast, test_series)
    mae = mae_loss(forecast, test_series)
    mape = mape_loss(forecast, test_series)
    return rmse, mae, mape

# 运行模型并评估数据集
def run_model(df):
    results = []
    for column in df.columns:  # 处理所有列
        series = df[column].dropna() # 归一化并去掉缺失值
        if series.empty:
            continue
        series = normalize_data(series)
        train_size = int(len(series) * 0.8)  # 80%作为训练集
        train_series = series[:train_size]
        test_series = series[train_size:]
        rmse, mae, mape = evaluate_model(train_series, test_series)
        results.append((column, rmse, mae, mape))
    return pd.DataFrame(results, columns=['Column', 'RMSE', 'MAE', 'MAPE'])

# 读取数据集并运行模型
df1 = pd.read_excel('data/EIA24/ARIMA/EIA24_year.xlsx')
results1 = run_model(df1)

print("\n第二个数据集\n")
df2 = pd.read_excel('data/EIA24/ARIMA/EIA24_STL_year.xlsx')
results2 = run_model(df2)

# 输出结果
print("EIA24 RMSE：", results1['RMSE'].mean(), "MAE：", results1['MAE'].mean(), "MAPE：", results1['MAPE'].mean())
print("EIA24_STL RMSE：", results2['RMSE'].mean(), "MAE：", results2['MAE'].mean(), "MAPE：", results2['MAPE'].mean())

# 保存结果到Excel文件
with pd.ExcelWriter('data/EIA24/ARIMA/results_year.xlsx') as writer:
    results1.to_excel(writer, sheet_name='EIA24', index=False)
    results2.to_excel(writer, sheet_name='EIA24_STL', index=False)

# RMSE对比图
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(results1['Column']))

plt.bar(index, results1['RMSE'], bar_width, label='EIA24', color='#B8D4E9')
plt.bar(index + bar_width, results2['RMSE'], bar_width, label='EIA24_STL', color='#2F7DBB')

plt.xlabel('Column')
plt.ylabel('RMSE')
plt.title('RMSE Loss per Column for Two Data Sets')
plt.xticks(index + bar_width / 2, results1['Column'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('data/EIA24/ARIMA/EIA24_ARIMA_rmse_comparison_year.png')
plt.show()