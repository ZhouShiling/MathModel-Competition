import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_excel('data_interpolate.xlsx')

# 准备数据
df.set_index('年份', inplace=True)  # 设置年份为索引
regions = df.columns  # 获取所有地区

# 插值
df.interpolate(method='linear', axis=0, inplace=True)  # 使用线性插值填充缺失值


# 计算年均增长率
def calculate_growth_rate(df, region):
    start_year = df.index.min()
    end_year = df.index.max()
    n_years = end_year - start_year
    initial_value = df.loc[start_year, region]
    final_value = df.loc[end_year, region]

    growth_rate = (final_value / initial_value) ** (1 / n_years) - 1
    return growth_rate * 100  # 转换为百分比形式


growth_rates = {region: calculate_growth_rate(df, region) for region in regions}

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制柱状图
plt.figure(figsize=(14, 8))
plt.bar(growth_rates.keys(), growth_rates.values())
plt.title('各地区医疗卫生机构数年均增长率')
plt.xlabel('地区')
plt.ylabel('年均增长率 (%)')
plt.xticks(rotation=90)
plt.show()

# 输出年均增长率
for region, rate in growth_rates.items():
    print(f"{region}: {rate:.4f}%")
