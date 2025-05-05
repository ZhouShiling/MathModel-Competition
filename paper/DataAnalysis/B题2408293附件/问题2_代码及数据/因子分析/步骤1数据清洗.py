import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file = '湖南省'
# 加载数据
data = pd.read_excel(file + '.xlsx')

# 选择相关的列
cols = [f"职工平均工资(元)_{file}", f"职工工资总额(万元)_{file}", f"医院数(个)_{file}", f"医院人员数(人)_{file}",
        f"医院床位数(张)_{file}", f"医生数(人)_{file}", f"卫生人员数(人)_{file}", f"人口自然增长率(%)_{file}",
        f"人均GDP(元)_{file}", f"农村居民消费水平(元)_{file}", f"劳动者报酬(亿元)_{file}", f"居民消费水平(元)_{file}",
        f"城镇居民消费水平(元)_{file}", f"城镇登记失业人员数(万人)_{file}",
        f"城镇登记失业率(%)_{file}", f"常住人口(万人)_{file}", f"GDP(亿元)_{file}", f'各地区医疗卫生机构数(个)_{file}']
X = data[cols]

# 计算相关性矩阵
corr_matrix = X.corr()
print("相关性矩阵:\n", corr_matrix)

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制相关性矩阵的热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=cols, yticklabels=cols)
# plt.title('变量之间的相关性热力图')
# plt.savefig('correlation_heatmap.png')
# print("相关性热力图已保存为 correlation_heatmap.png。")

# 计算相关性矩阵的逆矩阵
inv_corr_matrix = np.linalg.inv(corr_matrix)
print("\n相关性矩阵的逆矩阵:\n", inv_corr_matrix)

# 手动计算特征值和特征向量来生成碎石图
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
idx = eigenvalues.argsort()[::-1]  # eigenvalues从大到小排序
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-')
plt.xlabel('成分编号')
plt.ylabel('特征值')
plt.title('碎石图')
plt.savefig('碎石图.png')

# 打印特征值以帮助确定要采用的因子数
print("特征值:\n", eigenvalues)

# 输出一些基本的描述性统计数据
X_describe = X.describe()
print("\n基本统计数据:\n", X_describe)

# 保存相关性矩阵和其逆到Excel文件
with pd.ExcelWriter('./分析结果.xlsx') as writer:
    pd.DataFrame(corr_matrix, index=cols, columns=cols).to_excel(writer, sheet_name='Correlation Matrix')
    pd.DataFrame(inv_corr_matrix, index=cols, columns=cols).to_excel(writer, sheet_name='Inverse Correlation Matrix')
