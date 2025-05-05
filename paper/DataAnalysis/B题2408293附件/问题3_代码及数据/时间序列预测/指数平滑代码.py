import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_excel('附件1 1990~2023年各地区医疗卫生机构数(个).xlsx')

# 设置平滑因子 alpha
alpha = 0.5

# 对数据框中的每列数据应用指数平滑
smoothed_data = data.ewm(alpha=alpha).mean()

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制原始数据和平滑后的数据，这里仅以北京市为例进行展示
plt.figure(figsize=(10, 5))
plt.plot(data['北京市'].index, data['北京市'].values, label='指数平滑前')
plt.plot(smoothed_data['北京市'].index, smoothed_data['北京市'].values, label='指数平滑后', color='red')
plt.title('指数平滑前后对比_北京市')
plt.xlabel('年份')
plt.ylabel('医疗卫生机构数')
plt.legend()
plt.show()

# 将平滑后的数据保存到新的Excel文件
smoothed_data.to_excel('指数平滑后的数据.xlsx')

print("指数平滑后的数据已保存到 '指数平滑后的数据.xlsx' 文件中")
