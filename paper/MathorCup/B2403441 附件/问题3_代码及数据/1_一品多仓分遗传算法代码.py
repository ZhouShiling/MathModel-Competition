import random

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# 读取数据
warehouses = pd.read_csv('附件3.csv', encoding='gbk')
category_assoc = pd.read_csv('附件4.csv', encoding='gbk')
category_info = pd.read_csv('附件5.csv', encoding='gbk')
future_inventory = pd.read_csv('future_inventory_predictions_350_categories1.csv', encoding='gbk')
future_sales = pd.read_csv('transformed_data.csv', encoding='gbk')

# 创建品类到索引的映射表，将品类编码映射到整数索引
category_list = pd.concat([category_assoc['品类1'], category_assoc['品类2']]).unique()
category_to_index = {category: idx for idx, category in enumerate(category_list)}
index_to_category = {v: k for k, v in category_to_index.items()}

# 初始化关联度矩阵
num_categories = len(category_to_index)
num_warehouses = len(warehouses)
assoc_matrix = np.zeros((num_categories, num_categories))

# 填充关联度矩阵
for _, row in category_assoc.iterrows():
    idx1 = category_to_index[row['品类1']]
    idx2 = category_to_index[row['品类2']]
    assoc_matrix[idx1, idx2] = row['关联度']
    assoc_matrix[idx2, idx1] = row['关联度']  # 矩阵是对称的

# 获取预测的库存和销量数据
future_inventory = future_inventory.groupby('category').sum()['predicted_inventory'].reindex(category_list).fillna(
    0).values
future_sales = future_sales.groupby('category').sum()['predicted_sales'].reindex(category_list).fillna(0).values

# 初始化DEAP的Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 注册工具箱
toolbox = base.Toolbox()

# 定义个体
toolbox.register("attr_int", random.randint, 0, num_warehouses - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int,
                 num_categories * 3)  # 每个品类分配3个仓库
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 定义评价函数
def evaluate(individual):
    individual_matrix = np.array(individual).reshape(num_categories, 3)  # 每个品类分配3个仓库

    # 计算目标函数
    objective = 0
    for i in range(num_categories):
        for j in range(num_categories):
            if i != j:
                objective += assoc_matrix[i, j] * (1 if individual_matrix[i, 0] == individual_matrix[j, 0] else 0)
                objective += assoc_matrix[i, j] * (1 if individual_matrix[i, 1] == individual_matrix[j, 1] else 0)
                objective += assoc_matrix[i, j] * (1 if individual_matrix[i, 2] == individual_matrix[j, 2] else 0)

    # 检查约束条件
    violations = 0
    for i in range(num_categories):
        unique_warehouses = len(np.unique(individual_matrix[i]))
        if unique_warehouses > 3:
            violations += unique_warehouses - 3

    return objective - violations,


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
for i in range(num_categories):
    warehouses_assigned = []
    for j in range(3):
        warehouse_id = warehouses['仓库'][best_ind[i * 3 + j]]
        warehouses_assigned.append(warehouse_id)
    results.append([category_list[i], warehouses_assigned])

results_df = pd.DataFrame(results, columns=['category', 'warehouses'])
results_df.to_csv('category_warehouse_allocation_balanced.csv', index=False)

print("优化完成，分仓方案已保存为 category_warehouse_allocation_balanced.csv")
