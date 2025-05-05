import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

file = '湖南省'
# 加载数据
data = pd.read_excel(file + '.xlsx')
# 加载数据
# data = pd.read_excel('./国家.xlsx')
cols = [f"医院数(个)_{file}",
        f"医院床位数(张)_{file}", f"医生数(人)_{file}", f"卫生人员数(人)_{file}", f"人口自然增长率(%)_{file}",
        f"人均GDP(元)_{file}", f"劳动者报酬(亿元)_{file}", f"居民消费水平(元)_{file}",
        f"城镇居民消费水平(元)_{file}", f"城镇登记失业人员数(万人)_{file}",
        f"城镇登记失业率(%)_{file}", f"常住人口(万人)_{file}", f"GDP(亿元)_{file}", f'各地区医疗卫生机构数(个)_{file}']
X = data[cols]

# 计算KMO和巴特利特检验
kmo_all, kmo_model = calculate_kmo(X)
bartlett = calculate_bartlett_sphericity(X)
print("KMO 检验值:", kmo_model)
print("Bartlett检验值:", bartlett[0], "\np-value:", bartlett[1])

# 初始化因子分析对象并进行拟合
fa = FactorAnalyzer(rotation=None, n_factors=X.shape[1])
fa.fit(X)

# 检查公因子方差
print("Communality (各变量的公因子方巜):\n", fa.get_communalities())

# 解释的总方差
print("\n特征值:\n", fa.get_eigenvalues()[0])
print("\n解释的总方差:\n",
      pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings', 'Proportion Var', 'Cumulative Var']))

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制碎石图
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(fa.get_eigenvalues()[0]) + 1), fa.get_eigenvalues()[0], 'o-')
plt.title('碎石图')
plt.xlabel('因子数')
plt.ylabel('特征值')
plt.grid(True)
plt.savefig('scree_plot.png')
print("碎石图已保存至 scree_plot.png。")

# 进行方差最大化旋转
fa_rotate = FactorAnalyzer(rotation='varimax', n_factors=2)
fa_rotate.fit(X)
print("\n旋转后的因子负荷矩阵:\n", fa_rotate.loadings_)

# 提取得分系数矩阵
factor_scores = fa_rotate.transform(X)
print("\n得分系数矩阵:\n", factor_scores)

# 保存结果到Excel文件
results_df = pd.DataFrame(fa_rotate.loadings_, columns=[f"Factor{i + 1}" for i in range(2)], index=cols)
with pd.ExcelWriter('./因子分析结果.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Rotated Factor Loadings')
    pd.DataFrame(factor_scores, columns=[f"Factor{i + 1}" for i in range(2)]).to_excel(writer,
                                                                                       sheet_name='Factor Scores')

print("所有分析结果已保存至 因子分析结果.xlsx")
