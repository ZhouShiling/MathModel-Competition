import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = '插补后的数据.xlsx'
data = pd.read_excel(file_path)

# 选择需要处理的列
columns_to_process = data.columns[1:]  # 假设第一列为年份，从第二列开始处理

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 创建一个函数来绘制时间序列图
def plot_time_series(column_name):
    plt.figure(figsize=(14, 7))
    plt.plot(data['年份'], data[column_name], marker='o')
    plt.title(f'医疗卫生机构数时间序列 - {column_name}')
    plt.xlabel('年份')
    plt.ylabel('医疗卫生机构数')
    plt.grid(True)
    plt.show()


# 绘制每个地区的图
for column in columns_to_process:
    plot_time_series(column)
