import pandas as pd

# 创建一个字典来存储所有的数据
data_dict = {}

# 文件列表
file_list = [
    "职工平均工资(元).xlsx",
    "职工工资总额(万元).xlsx",
    "医院数(个).xlsx",
    "医院人员数(人).xlsx",
    "医院床位数(张).xlsx",
    "医生数(人).xlsx",
    "卫生人员数(人).xlsx",
    "人口自然增长率(%).xlsx",
    "人均GDP(元).xlsx",
    "农村居民消费水平(元).xlsx",
    "劳动者报酬(亿元).xlsx",
    "居民消费水平(元).xlsx",
    "各地区医疗卫生机构数(个).xlsx",
    "城镇居民消费水平(元).xlsx",
    "城镇登记失业人员数(万人).xlsx",
    "城镇登记失业率(%).xlsx",
    "常住人口(万人).xlsx",
    "GDP(亿元).xlsx"
]

# 读取每个文件
for file in file_list:
    # 读取Excel文件
    df = pd.read_excel(file)

    # 获取文件名作为前缀
    prefix = file.split(".")[0]

    # 将年份列添加到字典中
    data_dict[f"{prefix}_year"] = df.iloc[:, 0]

    # 将其他列添加到字典中
    for col in df.columns[1:]:
        data_dict[f"{prefix}_{col}"] = df[col]

# 创建一个新的DataFrame
df_final = pd.DataFrame(data_dict)

# 输出DataFrame以检查
print(df_final.head())

# 将结果保存到新的Excel文件
df_final.to_excel("总数据.xlsx", index=False)
