import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

file = '得分系数矩阵'
# 加载数据
data = pd.read_excel(file + '.xlsx')

# 选择做回归分析的特征列
features = data[
    ['Factor1', 'Factor2']]
target = data['各地区医疗卫生机构数(个)']

# 添加截距项，准备进行回归分析
features = sm.add_constant(features)

# 使用statsmodels的OLS进行多元线性回归
model = sm.OLS(target, features).fit()

# 打印出模型的摘要信息
print(model.summary())

# 进行德宾-沃森测试来检查残差的自相关性
dw = sm.stats.stattools.durbin_watson(model.resid)
print(f"Durbin-Watson statistic: {dw}")

# 创建DataFrame来保存VIF结果
vif_data = pd.DataFrame()
vif_data["Feature"] = features.columns
vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
print(vif_data)

# 绘制实际GDP与预测GDP的散点图
plt.figure(figsize=(10, 6))
plt.scatter(target, model.fittedvalues, alpha=0.75)
plt.plot([target.min(), target.max()], [model.fittedvalues.min(), model.fittedvalues.max()], color='red')  # 线性拟合
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted GDP Comparison')
plt.grid(True)
plt.savefig('actual_vs_predicted_gdp_comparison.png')
