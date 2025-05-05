import pandas as pd
import numpy as np

# 加载数据
data = pd.read_excel('医院人员数(人)1.xlsx')

# 将 "--" 替换为 NaN
data.replace('--', np.nan, inplace=True)

# 显示缺失值的数量
print(data.isnull().sum())

# 选择需要处理的列
columns_to_process = data.columns[1:]  # 假设第一列为年份，从第二列开始处理

# # 方法1: 前向填充
# data_ffill = data.copy()
# data_ffill[columns_to_process] = data_ffill[columns_to_process].fillna(method='ffill')
#
# # 方法2: 后向填充
# data_bfill = data.copy()
# data_bfill[columns_to_process] = data_bfill[columns_to_process].fillna(method='bfill')
#
# # 方法3: 使用均值填充
# data_mean = data.copy()
# data_mean[columns_to_process] = data_mean[columns_to_process].fillna(data_mean.mean())

# 方法4: 使用线性插值
data_interpolate = data.copy()
data_interpolate[columns_to_process] = data_interpolate[columns_to_process].interpolate()

# 保存处理后的数据到Excel文件
# data_ffill.to_excel('data_ffill.xlsx', index=False)
# data_bfill.to_excel('data_bfill.xlsx', index=False)
# data_mean.to_excel('data_mean.xlsx', index=False)
data_interpolate.to_excel('医院人员数(人)2.xlsx', index=False)

# 查看处理结果
# print(data_ffill.head())
# print(data_bfill.head())
# print(data_mean.head())
print(data_interpolate.head())
