import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
file = '湖南省'
# 加载数据
data = pd.read_excel(file + '.xlsx')

# 选择做回归分析的特征列
features = data[
    [f"职工平均工资(元)_{file}", f"职工工资总额(万元)_{file}", f"医院数(个)_{file}", f"医院人员数(人)_{file}",
     f"医院床位数(张)_{file}", f"医生数(人)_{file}", f"卫生人员数(人)_{file}", f"人口自然增长率(%)_{file}",
     f"人均GDP(元)_{file}", f"农村居民消费水平(元)_{file}", f"劳动者报酬(亿元)_{file}", f"居民消费水平(元)_{file}",
     f"城镇居民消费水平(元)_{file}", f"城镇登记失业人员数(万人)_{file}",
     f"城镇登记失业率(%)_{file}", f"常住人口(万人)_{file}", f"GDP(亿元)_{file}", f'各地区医疗卫生机构数(个)_{file}']]
target = data[f'各地区医疗卫生机构数(个)_{file}']

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
