import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_excel('./湖南省.xlsx')

# 选择数值型列进行标准化
numeric_cols = data.select_dtypes(include=['number']).columns
scaler = StandardScaler()

# 对数值型数据进行标准化处理
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 显示处理后的数据前五行以确保标准化已正确应用
print("标准化后的数据前五行:")
print(data.head())

# 保存处理后的数据到Excel文件
data.to_excel('./标准化后的数据.xlsx', index=False)
print("数据已保存到Excel文件。")
