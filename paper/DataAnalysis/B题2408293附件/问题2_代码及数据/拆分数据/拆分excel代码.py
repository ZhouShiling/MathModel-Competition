import pandas as pd

# 定义原始数据文件路径
input_file = '总数据.xlsx'
# 定义输出文件路径
output_file = '1中国.xlsx'

# 读取原始数据
df = pd.read_excel(input_file)

# 初始列号
column_number = 33
# 结束列号
end_column_number = 594
# 步进值
step = 33
# 创建一个空DataFrame用于存储结果
result_df = pd.DataFrame()

# 循环处理每一列，直到达到594列
while column_number <= end_column_number:
    # 获取当前列的数据
    column_data = df.iloc[:, column_number - 1]
    # 将当前列的数据添加到结果DataFrame中
    result_df = pd.concat([result_df, column_data], axis=1)
    # 更新列号
    column_number += step

# 保存结果到新的Excel文件
result_df.to_excel(output_file, index=False)
