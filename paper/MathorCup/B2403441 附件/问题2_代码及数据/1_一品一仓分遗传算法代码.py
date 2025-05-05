import random

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# 读取数据
warehouses = pd.read_csv('附件3.csv', encoding='gbk')
category_info = pd.read_csv('附件5.csv', encoding='gbk')
inventory_predictions = pd.read_csv('future_inventory_predictions_350_categories1.csv', encoding='gbk')
sales_predictions = pd.read_csv('transformed_data.csv', encoding='gbk')

# 提前处理数据，将数据转换为字典形式以便快速索引
category_inv_dict = {
    row['category']: row['predicted_inventory'] for _, row in inventory_predictions.iterrows()
}
category_sales_dict = {
    row['category']: row['predicted_sales'] for _, row in sales_predictions.iterrows()
}
warehouse_capacity_dict = {
    row['仓库']: row['仓容上限'] for _, row in warehouses.iterrows()
}
warehouse_production_dict = {
    row['仓库']: row['产能上限'] for _, row in warehouses.iterrows()
}
warehouse_rent_dict = {
    row['仓库']: row['仓租日成本'] for _, row in warehouses.iterrows()

}

# 检查列名并重命名为英文以避免KeyError
if len(category_info.columns) == 3:
    category_info.columns = ['category', 'high - level_category', 'type']
elif len(category_info.columns) == 2:
    category_info.columns = ['category', 'high - level_category']
warehouses.columns = ['warehouse_id', 'rent_cost', 'capacity_limit', 'daily_max_out']
if len(inventory_predictions.columns) == 3:
    inventory_predictions.columns = ['category', 'month', 'predicted_inventory']
elif len(inventory_predictions.columns) == 2:
    inventory_predictions.columns = ['category', 'predicted_inventory']
if len(sales_predictions.columns) == 3:
    sales_predictions.columns = ['category', 'date', 'predicted_sales']
elif len(sales_predictions.columns) == 2:
    sales_predictions.columns = ['category', 'predicted_sales']

# 初始化DEAP的Creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 注册工具箱
toolbox = base.Toolbox()

# 定义个体
toolbox.register("indices", random.sample, range(len(warehouses)), min(len(category_info), len(warehouses)))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 定义评价函数
def evaluate(individual):
    objective = 0
    violations = 0
    capacity_usage = {w: 0 for w in warehouses['warehouse_id']}
    production_usage = {w: 0 for w in warehouses['warehouse_id']}

    # 计算目标函数
    for cat_idx, warehouse_id in enumerate(individual):
        cat = category_info['category'][cat_idx]
        inv_pred = category_inv_dict.get(cat)
        sales_pred = category_sales_dict.get(cat)
        cap_limit = warehouse_capacity_dict.get(warehouses['warehouse_id'][warehouse_id])
        daily_max = warehouse_production_dict.get(warehouses['warehouse_id'][warehouse_id])
        rent_cost = warehouse_rent_dict.get(warehouses['warehouse_id'][warehouse_id])

        if inv_pred is not None and cap_limit is not None:
            objective += inv_pred / cap_limit
        if sales_pred is not None and daily_max is not None:
            objective += sales_pred / daily_max
        if rent_cost is not None:
            objective += rent_cost

        # 更新容量使用情况
        if inv_pred is not None:
            capacity_usage[warehouses['warehouse_id'][warehouse_id]] += inv_pred

        # 更新生产使用情况
        if sales_pred is not None:
            production_usage[warehouses['warehouse_id'][warehouse_id]] += sales_pred

    # 检查约束条件
    for warehouse_id, used_cap in capacity_usage.items():
        if used_cap > warehouse_capacity_dict.get(warehouse_id):
            violations += 1

    for warehouse_id, used_prod in production_usage.items():
        if used_prod > warehouse_production_dict.get(warehouse_id):
            violations += 1

    return objective + violations,


# 注册评估函数
toolbox.register("evaluate", evaluate)

# 注册变异、交叉、选择操作
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 设置参数
POP_SIZE = 100
GENS = 50
CXPB = 0.7
MUTPB = 0.2

# 初始化种群
population = toolbox.population(n=POP_SIZE)

# 运行遗传算法
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

hall_of_fame = tools.HallOfFame(1)

# 运行遗传算法
result = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENS, stats=stats,
                             halloffame=hall_of_fame, verbose=True)

# 输出结果
best_ind = hall_of_fame[0]

# 构建结果DataFrame
results = []
for i in range(min(len(category_info), len(warehouses))):
    warehouse_id = warehouses['warehouse_id'][best_ind[i]]
    results.append([category_info['category'][i], warehouse_id])

results_df = pd.DataFrame(results, columns=['category', 'warehouse_id'])
results_df.to_csv('warehouse_allocation_results_ga.csv', index=False)

print("优化完成，分仓方案已保存为 warehouse_allocation_results_ga.csv")

