import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import time
# 数据载入
data = pd.read_excel('./smoothed_data.xlsx')

# 地区列表
names = [
    "中国", "北京市", "天津市", "河北省", "山西省", "内蒙古", "辽宁省", "吉林省",
    "黑龙江", "上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省",
    "山东省", "河南省", "湖北省", "湖南省", "广东省", "广西", "海南省",
    "重庆市", "四川省", "贵州省", "云南省", "西藏", "陕西省", "甘肃省",
    "青海省", "宁夏", "新疆"
]

# 遍历地区进行预测
for name in names:
    print(f'正在处理地区：{name}')

    # 使用auto_arima自动选择ARIMA模型的参数
    model = auto_arima(data[f'{name}'], seasonal=False, error_action='ignore')

    # 建立ARIMA模型并拟合数据
    try:
        model_fit = model.fit(data[f'{name}'])

        # 打印模型摘要
        print(model_fit.summary())

        # 预测未来5个时期的数据
        forecast = model_fit.predict(n_periods=5)
        print('预测结果:\n', [f'{val:f}' for val in forecast])
        print("best model：", model)  # 拟合出来最佳模型
        forecast_df = pd.DataFrame({
            'Forecast': forecast,
            'Year': [data.index[-1] + i + 1 for i in range(5)]  # 这里假设数据的索引是年份
        })

        # 保存预测结果到Excel文件
        forecast_df.to_excel(f'预测结果_{name}.xlsx', index=False)

        print('预测结果已保存到Excel文件')

        plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        # 绘制原始数据及预测数据
        plt.figure(figsize=(10, 6))
        plt.plot(data[f'{name}'], label='原始数据')
        plt.plot(forecast_df['Year'], forecast_df['Forecast'], label='预测数据', color='red')
        plt.title(f'医疗卫生机构数预测_{name}')
        plt.xlabel('年份')
        plt.ylabel('医疗卫生机构数预测')
        plt.legend()
        plt.savefig(f'医疗卫生机构数预测_{name}.png')
        time.sleep(1)

    except ValueError as e:
        print(f'模型拟合错误({name}):', e)

    plt.close()  # 关闭绘图窗口，避免多次绘图冲突
