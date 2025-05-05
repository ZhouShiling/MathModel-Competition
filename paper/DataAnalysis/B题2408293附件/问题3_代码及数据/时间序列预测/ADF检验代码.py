import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 从Excel文件读取数据
df = pd.read_excel('smoothed_data.xlsx')
data = df['中国'].values

# 滞后阶数列表
lag_orders = [1, 2, 3]

# 循环检验不同的滞后阶数
for lag in lag_orders:
    # 执行ADF检验
    result = adfuller(data, maxlag=lag, autolag=None)

    # 提取ADF检验结果
    adf_statistic = result[0]  # ADF 统计量
    p_value = result[1]  # p值
    used_lag = result[2]  # 使用的滞后阶数
    critical_values = result[4]  # 临界值

    # 打印结果
    print(f"\n滞后阶数: {lag}")
    print(f"ADF 统计量: {adf_statistic}")
    print(f"p值: {p_value}")
    print(f"使用的滞后阶数: {used_lag}")
    # print(f"AIC: {result[5]}")  # 自动选择的滞后阶数对应的AIC

    print("\n临界值：")
    for key, value in critical_values.items():
        print(f"{key} 临界值: {value}")
