import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('Q32相关分析原始数据.xlsx', sheet_name='Sheet1')

# 提取感兴趣的列
crop_data = df[['亩产量/斤', '种植成本/(元/亩)', '平均销售单价/(元/斤)']]

# 计算相关系数矩阵
correlation_matrix = crop_data.corr()

# 设置绘图风格
sns.set(style="white")

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 创建热力图
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})

# 显示图形
plt.title('作物亩产量、种植成本与销售单价相关性分析')
plt.show()
