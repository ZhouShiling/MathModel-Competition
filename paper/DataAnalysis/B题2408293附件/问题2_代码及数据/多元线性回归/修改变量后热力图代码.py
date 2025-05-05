import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file = '湖南省'
# 加载数据
data = pd.read_excel(file + '.xlsx')

# 选择特征和目标变量
features_with_target = data[
    [f"医院数(个)_{file}",
     f"医院床位数(张)_{file}", f"医生数(人)_{file}", f"卫生人员数(人)_{file}", f"人口自然增长率(%)_{file}",
     f"人均GDP(元)_{file}", f"劳动者报酬(亿元)_{file}", f"居民消费水平(元)_{file}",
     f"城镇居民消费水平(元)_{file}", f"城镇登记失业人员数(万人)_{file}",
     f"城镇登记失业率(%)_{file}", f"常住人口(万人)_{file}", f"GDP(亿元)_{file}", f'各地区医疗卫生机构数(个)_{file}']]

# 计算相关系数矩阵
correlation_matrix = features_with_target.corr()

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制热力图
plt.figure(figsize=(30, 20))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, linewidths=.5)
plt.title('变量间的相关系数矩阵热力图')
plt.savefig(f'{file}correlation_matrix_heatmap.png')
plt.close()
