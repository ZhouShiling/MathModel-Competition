import matplotlib.pyplot as plt
import pandas as pd
import time

# Load the data into a pandas DataFrame.
data = pd.read_excel('./附件1 1990~2023年各地区医疗卫生机构数(个).xlsx')

# 地区列表
names = [
    "中国", "北京市", "天津市", "河北省", "山西省", "内蒙古", "辽宁省", "吉林省",
    "黑龙江", "上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省",
    "山东省", "河南省", "湖北省", "湖南省", "广东省", "广西", "海南省",
    "重庆市", "四川省", "贵州省", "云南省", "西藏", "陕西省", "甘肃省",
    "青海省", "宁夏", "新疆"
]

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 创建一个字典来存储每个地区的一阶差分数据
diff_data = {}

# 对于每个地区，计算一阶差分
for name in names:
    diff_data[name] = data[name].diff()

# 绘制每个地区的一阶差分时间序列图
for name in names:
    plt.figure(figsize=(12, 6))
    plt.plot(data['年份'].iloc[1:], diff_data[name].iloc[1:], marker='o', linestyle='-', label=name)
    plt.title(f'{name} 一阶差分的时间序列图')
    plt.xlabel('年份')
    plt.ylabel('一阶差分')
    plt.legend()
    plt.grid(True)
    time.sleep(1)
    plt.savefig(f'{name}差分序列图.png')
    print(f'{name}差分序列图已保存')
