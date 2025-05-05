import matplotlib.pyplot as plt
import pandas as pd

# Load the data into a pandas DataFrame.
data = pd.read_excel('./附件1 1990~2023年各地区医疗卫生机构数(个).xlsx')

name = '新疆'

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# Plotting the fertilizer consumption over time separately
plt.figure(figsize=(12, 6))
plt.plot(data['年份'], data[f'{name}'], marker='o', linestyle='-', color='b')
plt.title(f'{name}医疗卫生机构数随时间的变化')

plt.xlabel('年份')
plt.ylabel('医疗卫生机构数(个)')
plt.grid(True)
plt.savefig(f'{name}医疗卫生机构数随时间的变化')

