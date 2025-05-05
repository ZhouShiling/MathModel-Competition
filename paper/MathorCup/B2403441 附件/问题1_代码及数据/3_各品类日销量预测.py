import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gm11(x0, predict_num):
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2

    y = x0[1:]
    x = z1

    k = ((n - 1) * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
        ((n - 1) * np.sum(x ** 2) - np.sum(x) ** 2)
    b = (np.sum(x ** 2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / \
        ((n - 1) * np.sum(x ** 2) - np.sum(x) ** 2)
    a = -k

    x0_hat = np.zeros(n)
    x0_hat[0] = x0[0]
    for m in range(1, n):
        x0_hat[m] = (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * m)

    result = np.zeros(predict_num)
    for i in range(predict_num):
        result[i] = (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * (n + i))

    return result


def metabolism_gm11(x0, predict_num):
    result = np.zeros(predict_num)
    for i in range(predict_num):
        res = gm11(x0, 1)[0]
        result[i] = res
        x0 = np.concatenate((x0[1:], [res]))  # Update x0 with the predicted value

    return result


def new_gm11(x0, predict_num):
    result = np.zeros(predict_num)
    for i in range(predict_num):
        res = gm11(x0, 1)[0]
        result[i] = res
        x0 = np.append(x0, res)  # Append the predicted value to the end of x0

    return result




# 加载数据，假设文件编码为 gbk
file_path = '附件2.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 检查列名称
print("数据列名：", data.columns)

# 转换'日期'列为日期格式
data['日期'] = pd.to_datetime(data['日期'], format='%Y/%m/%d')

# 创建存储未来三个月预测结果的 DataFrame
future_predictions_all = pd.DataFrame()

# 逐个品类进行训练和预测
for category in data['品类'].unique():
    # 选择该品类的数据
    category_data = data[data['品类'] == category]

    # 检查样本量，如果样本量不足，则跳过该品类
    if len(category_data) < 2:
        print(f"品类 {category} 样本量不足，跳过训练")
        continue

    # 提取销量数据
    sales_data = category_data['销量'].values

    # 进行灰色预测
    # 这里假设 gm11, new_gm11 和 metabolism_gm11 函数已经定义好
    predicted_sales = gm11(sales_data, 92)  # 替换为合适的灰色模型函数

    # 生成未来日期
    last_date = category_data['日期'].max()
    future_dates = pd.date_range(start=pd.to_datetime('7/1/2024'), end=pd.to_datetime('9/30/2024'))

    # 构建该品类的预测结果 DataFrame
    future_predictions = pd.DataFrame({
        'category': [category] * len(future_dates),
        'date': future_dates,
        'predicted_sales': predicted_sales
    })

    # 将该品类的预测结果添加到总的 DataFrame 中
    future_predictions_all = pd.concat([future_predictions_all, future_predictions], axis=0)

# 保存未来三个月每天的销量预测结果到 CSV 文件
future_predictions_all.to_csv('future_daily_sales_predictions_350_categories_gm11.csv', index=False)

# 输出最终结果
print(future_predictions_all)
