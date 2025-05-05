import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import time

# 读取Excel数据
data = pd.read_excel('./附件1 1990~2023年各地区医疗卫生机构数(个).xlsx')

# 地区列表
names = [
    "中国", "北京市", "天津市", "河北省", "山西省", "内蒙古", "辽宁省", "吉林省",
    "黑龙江", "上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省",
    "山东省", "河南省", "湖北省", "湖南省", "广东省", "广西", "海南省",
    "重庆市", "四川省", "贵州省", "云南省", "西藏", "陕西省", "甘肃省",
    "青海省", "宁夏", "新疆"
]

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False

# 重命名列名以匹配数据集
data.rename(columns={'年份': 'year', '医疗卫生机构数量': 'China'}, inplace=True)

# 设置索引为 'year'
data.set_index('year', inplace=True)

# 循环遍历每个地区
for name in names:
    if name not in data.columns:
        continue

    # 计算一阶差分
    data[f'{name}_diff'] = data[name].diff()

    # 绘制ACF和PACF图
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    sm.graphics.tsa.plot_acf(data[f'{name}_diff'].dropna(), lags=15, ax=ax[0],
                             title=f'{name} 医疗卫生机构数量一阶差分的自相关图')
    sm.graphics.tsa.plot_pacf(data[f'{name}_diff'].dropna(), lags=15, ax=ax[1],
                              title=f'{name} 医疗卫生机构数量一阶差分的偏自相关图')
    plt.tight_layout()
    plt.savefig(f'{name}自相关、偏自相关图.png')
    print(f'{name}自相关、偏自相关图已保存')
    time.sleep(1)
    plt.close(fig)  # 关闭图像以释放内存
