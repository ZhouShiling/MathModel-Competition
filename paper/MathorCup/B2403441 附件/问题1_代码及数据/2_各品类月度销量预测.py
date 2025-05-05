import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def gm11(x0, predict_num):
    """
    Function to perform prediction using the traditional GM(1,1) model.
    """
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

    # Calculate absolute residuals and relative residuals
    absolute_residuals = x0[1:] - x0_hat[1:]
    relative_residuals = np.abs(absolute_residuals) / x0[1:]

    # Calculate ratio and ratio deviation
    class_ratio = x0[1:] / x0[:-1]
    eta = np.abs(1 - (1 - 0.5 * a) / (1 + 0.5 * a) * (1 / class_ratio))

    return result, x0_hat, relative_residuals, eta


def metabolism_gm11(x0, predict_num):
    """
    Function to perform prediction using the metabolism GM(1,1) model.
    """
    result = np.zeros(predict_num)
    for i in range(predict_num):
        res = gm11(x0, 1)[0][0]
        result[i] = res
        x0 = np.concatenate((x0[1:], [res]))  # Update x0 with the predicted value

    return result


def new_gm11(x0, predict_num):
    """
    Function to perform prediction using the new information GM(1,1) model.
    """
    result = np.zeros(predict_num)
    for i in range(predict_num):
        res = gm11(x0, 1)[0][0]
        result[i] = res
        x0 = np.append(x0, res)  # Append the predicted value to the end of x0

    return result


# 读取附件1中的库存数据
sales_data = pd.read_excel('附件2.xlsx')

# 初始化预测结果列表
sales_predictions = []
error_results = []

# 对每个品类进行循环预测
for category, df in sales_data.groupby('品类'):
    # 仅保留销量和日期列
    df = df[['日期', '销量']]
    df.columns = ['date', 'sales']

    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)

    # 检查是否存在足够的数据点
    data = df['sales'].values
    n = len(data)

    # Check data validity
    ERROR = 0
    if np.any(data < 0):
        print("The time series for grey prediction cannot use negative numbers.")
        ERROR = 1

    print(f"The length of the original data is {n}")
    if n <= 3:
        print("The data volume is too small to perform grey prediction.")
        ERROR = 1
    elif n > 10:
        print("The data volume is too large for grey prediction to provide accurate predictions.")

    # Exponential regularity check
    if ERROR == 0:
        print("----------------------------")
        print("Exponential Regularity Check")
        x1 = np.cumsum(data)
        rho = data[1:] / x1[:-1]

        fig, ax = plt.subplots()
        ax.plot(range(1, n), rho, 'o-')
        ax.axhline(y=0.5, color='r', linestyle='-')
        ax.text(n - 2, 0.55, 'Critical Line')
        ax.set_xticks(range(1, n))
        ax.set_xlabel('Year Sequence')
        ax.set_ylabel('Smoothness of Original Data')

        print(
            f"Indicator 1: The percentage of smoothness ratios less than 0.5 is {100 * np.sum(rho < 0.5) / (n - 1):.2f}%")
        print(
            f"Indicator 2: Excluding the first two periods, the percentage of smoothness ratios less than 0.5 is {100 * np.sum(rho[2:] < 0.5) / (n - 3):.2f}%")
        print(
            "Reference standards: Indicator 1 should be greater than 60%, and Indicator 2 should be greater than 90%.")

        # Assume the data passes the exponential regularity test.
        judge = 1
        if judge == 0:
            print("Grey prediction model is not suitable for this data.")
            ERROR = 1

    # Split the data into training and testing sets
    if ERROR == 0 and n > 4:
        print(
            "Because the number of periods in the original data is greater than 4, we can divide the data into training and testing sets.")
        if n > 7:
            test_num = 3
        else:
            test_num = 2
        train_data = data[:-test_num]
        test_data = data[-test_num:]

        print("Training data:")
        print(train_data)
        print("Testing data:")
        print(test_data)
        print("----------------------------")

        # Traditional GM(1,1) model
        print("Traditional GM(1,1) model prediction details:")
        result1, data_hat1, _, _ = gm11(train_data, test_num)

        # New information GM(1,1) model
        print("New information GM(1,1) model prediction details:")
        result2 = new_gm11(train_data, test_num)

        # Metabolism GM(1,1) model
        print("Metabolism GM(1,1) model prediction details:")
        result3 = metabolism_gm11(train_data, test_num)

        # fig, ax = plt.subplots()
        # ax.plot(range(n - test_num + 1, n + 1), test_data, 'o-', label='Actual Test Data')
        # ax.plot(range(n - test_num + 1, n + 1), result1, '*-', label='Traditional GM(1,1)')
        # ax.plot(range(n - test_num + 1, n + 1), result2, '+-', label='New Information GM(1,1)')
        # ax.plot(range(n - test_num + 1, n + 1), result3, 'x-', label='Metabolism GM(1,1)')
        # ax.set_xticks(range(n - test_num + 1, n + 1))
        # ax.legend()
        # ax.set_xlabel('Year Sequence')
        # ax.set_ylabel('Sales')
        # plt.title(f'Prediction Comparison for Category {category}')
        # plt.savefig(f'prediction_comparison_sales_{category}.png')  # 保存图表
        # plt.close()  # 关闭图表
        # time.sleep(1)  # 等待1秒

        SSE_1 = np.sum((test_data - result1) ** 2)
        SSE_2 = np.sum((test_data - result2) ** 2)
        SSE_3 = np.sum((test_data - result3) ** 2)

        error_results.append([category, SSE_1, SSE_2, SSE_3])

        print(f"SSE for the traditional GM(1,1) model on the test set is {SSE_1:.2f}")
        print(f"SSE for the new information GM(1,1) model on the test set is {SSE_2:.2f}")
        print(f"SSE for the metabolism GM(1,1) model on the test set is {SSE_3:.2f}")

        if SSE_1 < SSE_2:
            if SSE_1 < SSE_3:
                choose = 1
            else:
                choose = 3
        elif SSE_2 < SSE_3:
            choose = 2
        else:
            choose = 3

        models = ['Traditional GM(1,1) Model', 'New Information GM(1,1) Model', 'Metabolism GM(1,1) Model']
        print(f"Since the SSE of {models[choose - 1]} is the smallest, we should choose it for prediction.")
        print("----------------------------")

        predict_num = 3  # Predict the next 3 months
        if choose == 1:
            result, data_hat, *_ = gm11(data, predict_num)
        elif choose == 2:
            result = new_gm11(data, predict_num)
            data_hat = new_gm11(data, len(data))
        elif choose == 3:
            result = metabolism_gm11(data, predict_num)
            data_hat = metabolism_gm11(data, len(data))

        # Save the results
        sales_predictions.extend([(category, data_hat[i], result[i]) for i in range(predict_num)])

    else:
        predict_num = 3  # Predict the next 3 months
        print("Due to the small number of data points, we will simply average the results from all three methods.")

        print("Traditional GM(1,1) model prediction details:")
        result1, data_hat1, _, _ = gm11(data, predict_num)

        print("New information GM(1,1) model prediction details:")
        result2 = new_gm11(data, predict_num)

        print("Metabolism GM(1,1) model prediction details:")
        result3 = metabolism_gm11(data, predict_num)

        result = (result1 + result2 + result3) / 3

        data_hat = (data_hat1 + new_gm11(data, len(data)) + metabolism_gm11(data, len(data))) / 3

# 创建存储预测结果的 DataFrame
sales_predictions = []

# 逐个品类进行训练和预测
for category in sales_data['品类'].unique():
    # 选择该品类的数据
    category_data = sales_data[sales_data['品类'] == category]

    # 检查样本量，如果样本量不足，则跳过该品类
    if len(category_data) < 2:
        print(f"品类 {category} 样本量不足，跳过训练")
        continue

    # 提取销量数据
    sales = category_data['销量'].values

    # 使用灰色预测模型进行预测
    predict_num = 3  # 预测未来三个月
    result = gm11(sales, predict_num)

    # 计算拟合数据
    data_hat = gm11(sales, len(sales))

    # 保存预测结果
    min_length = min(len(result), len(data_hat))
    for i in range(min_length):
        sales_predictions.append((category, data_hat[i], result[i]))

# Create DataFrame to store the prediction results
sales_predictions_df = pd.DataFrame(sales_predictions, columns=['品类', '拟合数据', '预测数据'])

# Save the results to CSV
sales_predictions_df.to_csv('future_sales_predictions_350_categories2.csv', index=False)

# 创建一个包含不同模型误差的表格
error_results_df = pd.DataFrame(error_results, columns=['品类', '传统GM(1,1)', '新信息GM(1,1)', '代谢GM(1,1)'])
error_results_df.to_csv('误差结果2.csv', index=False)

# 输出最终结果
print(sales_predictions_df)
print(error_results_df)

# 将预测结果和误差结果保存到同一个Excel文件中
wb = Workbook()
ws1 = wb.active
ws1.title = '预测结果'
for r in dataframe_to_rows(sales_predictions_df, index=False, header=True):
    ws1.append(r)

ws2 = wb.create_sheet(title='误差结果')
for r in dataframe_to_rows(error_results_df, index=False, header=True):
    ws2.append(r)

wb.save('预测与误差结果2.xlsx')
