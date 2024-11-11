import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt


def dfa(time_series, min_window=2, max_window=1000):
    n = len(time_series)
    mean = np.mean(time_series)
    y = np.cumsum(time_series - mean)

    scales = []
    stds = []

    for window_size in range(min_window, max_window + 1):
        n_windows = n // window_size
        std_window = []

        for i in range(n_windows):
            segment = y[i * window_size:(i + 1) * window_size]
            x = np.arange(window_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            std_window.append(np.std(detrended))

        scales.append(window_size)
        stds.append(np.mean(std_window))

    coeffs = np.polyfit(np.log(scales), np.log(stds), 1)
    alpha = coeffs[0]

    return alpha



df = pd.read_excel('F:/AAA/GNN_Gaussian/data/ENTSO/ENTSO_filled.xlsx')


final_data = []
reshaped_columns = []

# 处理每一列
for j in range(df.shape[1]):
    series = df.iloc[:, j].dropna().values

    print(f"Processing Column {j} - Length: {len(series)}, Values: {series}")

    stl = STL(series, period=15400)
    result = stl.fit()
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    alpha_seasonal = dfa(seasonal)
    print(f"column:{j},alpha:{alpha_seasonal}")

    if 1 <= alpha_seasonal <= 2:
        final_data.append(series)
    else:
        new_data = trend + residual
        final_data.append(new_data)
        reshaped_columns.append(j)

    # 绘制STL分解结果
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(series, label='Original Series')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual')
    plt.legend(loc='upper left')
    plt.suptitle(f'STL Decomposition for Column {j}')
    plt.tight_layout()
    plt.savefig(f'F:/AAA/GNN_Gaussian/data/ENTSO/STL/15400/stl_15400_decomposition_column_{j}.png')
    plt.close()  # 关闭图形以释放内存


# 将结果保存为新的数据框
final_df = pd.DataFrame(final_data).T
final_df.columns = df.columns
final_df.to_excel('F:/AAA/GNN_Gaussian/data/ENTSO/STL/15400/final_data_15400.xlsx', index=False)


# 处理每一列
for j in range(final_df.shape[1]):
    series = final_df.iloc[:, j].dropna().values

    print(f"Processing Column {j} - Length: {len(series)}, Values: {series}")

    stl = STL(series, period=15400)
    result = stl.fit()
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    alpha_seasonal = dfa(seasonal)
    print(f"column:{j},alpha:{alpha_seasonal}")

    # 绘制STL分解结果
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(series, label='Original Series')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual')
    plt.legend(loc='upper left')
    plt.suptitle(f'STL Decomposition for Column {j}')
    plt.tight_layout()
    plt.savefig(f'F:/AAA/GNN_Gaussian/data/ENTSO/STL/15400/stl_15400_column_{j}.png')
    plt.close()  # 关闭图形以释放内存



# 输出重新整合的列信息
for column_index in reshaped_columns:
    print(f"Column {column_index} was reshaped.")

print("最终数据集已保存为 final_data.xlsx")
