# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from semopy import Model

# 数据预处理
df = pd.read_excel('研究生AI能力、因素、学习适应、科研效能感.xlsx')

# 列名简化处理（根据实际问卷结构对应）
column_mapping = {
    # 人本思维能力（9题）
    '9、人本思维能力—1.在实践中，我能够主动选择并使用数字工具': 'Q1',
    '9、2.我对进行AI技术的学习有积极的态度': 'Q2',
    '9、3.我需要通过AI技术提升学习与科研能力': 'Q3',
    '9、4.您认为在教育中由人工智能代替您做出决策的程度': 'Q4',
    '9、5.当使用人工智能时，我能够评估AI在学习决策中的潜在风险': 'Q5',
    '9、6.当使用人工智能时，我能够评估AI在学习决策中的合适性': 'Q6',
    '9、7.我认为AI技术能够对我的行为规范产生影响': 'Q7',
    '9、8.我认为AI技术能够对我的社会价值观产生影响': 'Q8',
    '9、9.我认为AI技术的使用能够实现平等受益': 'Q9',

    # 人工智能伦理能力（9题）
    '10、人工智能伦理能力—1.在选择AI生成的内容时，我能够关注其可靠性和道德性': 'Q10',
    '10、2.我了解使用AI技术的安全和隐私保护的重要性': 'Q11',
    '10、3.我在使用AI进行科研时，能关注到学术伦理': 'Q12',
    '10、4.在使用AI技术时，我能够尊重与保护数据隐私与产权': 'Q13',
    '10、5.在使用AI技术时，我能够遵守法律法规': 'Q14',
    '10、6.在使用AI生成内容时，我能够带有批判性的甄别': 'Q15',
    '10、7.我认为人工智能技术在教育中的应用，能够被大众理解': 'Q16',
    '10、8.我能够在使用AI技术的过程中明确责任。尊重规则': 'Q17',
    '10、9.在进行学术数据搜集时，我能够确保个人数据匿名化，符合规范': 'Q18',

    # 人工智能基础和应用能力（9题）
    '11、人工智能基础和应用能力—1.我能够掌握基本的人工智能概念': 'Q19',
    '11、2.我能够掌握基本的人工智能技术': 'Q20',
    '11、3.我了解基本的AI智能指令': 'Q21',
    '11、4.我能够利用AI技术促进数字学习': 'Q22',
    '11、5.我能够利用AI技术促进专业学习': 'Q23',
    '11、6.我能够利用AI技术促进学术科研': 'Q24',
    '11、7.我能够利用AI技术提升学习能力': 'Q25',
    '11、8.我能够利用AI技术优化学习环境': 'Q26',
    '11、9.我能够利用AI技术创新科研模式': 'Q27',

    # 人工智能学习能力（12题）
    '12、人工智能学习能力—1.我能够利用AI辅助工具设定学习目标': 'Q28',
    '12、2.我能够利用AI辅助工具选择学习内容': 'Q29',
    '12、3.我能够利用AI辅助工具优化学习效果': 'Q30',
    '12、4.我能够利用AI技术整合数字学习资源': 'Q31',
    '12、5.我能够利用AI技术整合数字学习路径': 'Q32',
    '12、6.我能够利用AI技术整合数字资源完善个性化学习': 'Q33',
    '12、7.我能够利用AI技术创建新型学习模式': 'Q34',
    '12、8.我能够利用AI技术调整学习策略，满足个性化学习需求': 'Q35',
    '12、9.我能够利用AI技术满足自身在特定及跨学科学习中的需求': 'Q36',
    '12、10.我能够利用AI技术优化知识获取模式': 'Q37',
    '12、11.我能够利用AI技术生成知识网络': 'Q38',
    '12、12.我能够利用AI技术实现知识的转化': 'Q39'
}
df = df.rename(columns=column_mapping)

# ------------------------- 模型定义 -------------------------
model_spec = '''
# 测量模型
人本思维能力 =~ Q1 + Q2 + Q3 + Q4 + Q5 + Q6 + Q7 + Q8 + Q9
人工智能伦理能力 =~ Q10 + Q11 + Q12 + Q13 + Q14 + Q15 + Q16 + Q17 + Q18
人工智能基础应用能力 =~ Q19 + Q20 + Q21 + Q22 + Q23 + Q24 + Q25 + Q26 + Q27
人工智能学习能力 =~ Q28 + Q29 + Q30 + Q31 + Q32 + Q33 + Q34 + Q35 + Q36 + Q37 + Q38 + Q39

# 结构模型
人工智能基础应用能力 ~ 人本思维能力
人工智能学习能力 ~ 人工智能基础应用能力 + 人本思维能力
人工智能伦理能力 ~ 人本思维能力 + 人工智能基础应用能力

# 协方差
人本思维能力 ~~ 人工智能伦理能力
'''

# ------------------------- 模型拟合 -------------------------
model = Model(model_spec)
# model.fit(df, obj='MLW', solver='SLSQP', tol=1e-8)
# 原始代码替换为：
fit_result = model.fit(df, obj='MLW', solver='SLSQP', tol=1e-8)  # 返回SolverResult对象
# ------------------------- 参数估计 -------------------------
param_df = model.inspect()

# 转换数值列
numeric_cols = ['Estimate', 'Std. Err', 'z-value', 'p-value']
param_df[numeric_cols] = param_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
param_df.fillna({'p-value': 1.0}, inplace=True)  # 填充缺失值

print("参数估计：\n", param_df)

# ------------------------- 拟合指标 -------------------------
fit_indices = model.fit()
print("\n拟合指标：\n", fit_indices)

# ------------------------- 标准化解 -------------------------
std_solution = param_df.copy()
std_solution['Estimate'] = std_solution['Estimate'] / std_solution['Std. Err']
print("\n标准化解：\n", std_solution)

# ------------------------- 可视化设置 -------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建网络图（只包含潜变量）
G = nx.DiGraph()
latent_vars = ['人本思维能力', '人工智能伦理能力', '人工智能基础应用能力', '人工智能学习能力']

# 节点配置
node_config = {
    "人本思维能力": {"color": "lightgreen", "type": "外生变量"},
    "人工智能伦理能力": {"color": "skyblue", "type": "中介变量"},
    "人工智能基础应用能力": {"color": "skyblue", "type": "中介变量"},
    "人工智能学习能力": {"color": "lightcoral", "type": "结果变量"}
}

# 添加潜变量节点
for var in latent_vars:
    G.add_node(var, **node_config[var])

# 添加结构路径和协方差
for _, row in param_df.iterrows():
    if row['op'] == '~':
        source = row['rval']
        target = row['lval']
        if source in latent_vars and target in latent_vars:  # 确保只添加潜变量
            weight = row['Estimate']
            pval = row['p-value']

            # 显著性标记
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            label = f"{weight:.2f}{stars}"

            # 边属性
            edge_attrs = {
                'color': 'blue' if weight > 0 else 'red',
                'width': abs(weight) * 4,
                'label': label,
                'connectionstyle': 'arc3,rad=0.2'  # 默认连接样式
            }
            G.add_edge(source, target, **edge_attrs)

    elif row['op'] == '~~':
        # 协方差（双箭头）
        if row['lval'] in latent_vars and row['rval'] in latent_vars:  # 确保只添加潜变量
            G.add_edge(row['lval'], row['rval'],
                       color='purple',
                       label=f"{row['Estimate']:.2f}",
                       connectionstyle="arc3,rad=0.3",
                       width=2)

# ------------------------- 绘制图形 -------------------------
plt.figure(figsize=(14, 10))

# 手动调整节点位置
pos = {
    '人本思维能力': (0, 1),
    '人工智能基础应用能力': (1, 1),
    '人工智能伦理能力': (0, 0),
    '人工智能学习能力': (1, 0)
}

# 绘制节点
node_colors = [node_config[n]['color'] for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, alpha=0.9)

# 绘制边
edges = G.edges(data=True)
edge_colors = [data['color'] for _, _, data in edges]
edge_widths = [data['width'] for _, _, data in edges]
connection_styles = [data.get('connectionstyle', 'arc3,rad=0.2') for _, _, data in edges]  # 默认连接样式

nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_colors,
    width=edge_widths,
    arrowsize=25,
    arrowstyle='->',
    connectionstyle=connection_styles
)

# 添加标签
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
edge_labels = {(u, v): data['label'] for u, v, data in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='black')

# 创建图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='外生变量',
               markersize=15, markerfacecolor='#4CAF50'),
    plt.Line2D([0], [0], marker='o', color='w', label='中介变量',
               markersize=15, markerfacecolor='#2196F3'),
    plt.Line2D([0], [0], marker='o', color='w', label='结果变量',
               markersize=15, markerfacecolor='#FF5252'),
    plt.Line2D([0], [0], color='#4CAF50', lw=3, label='正向效应'),
    plt.Line2D([0], [0], color='#FF5252', lw=3, label='负向效应'),
    plt.Line2D([0], [0], color='#9C27B0', lw=3, label='协方差关系')
]

# 调整图例位置
legend = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, borderaxespad=0.)

# 添加背景网格
plt.grid(True, linestyle='--', alpha=0.2)

plt.title("研究生AI能力结构方程模型路径图（潜变量层级）", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(right=0.8)  # 调整图像右侧留白
plt.show()


# ------------------------- 修改后的结果导出函数 -------------------------
def save_sem_results(param_df, fit_obj, std_solution, filename):
    """结构化保存SEM分析结果到Excel"""
    # 转换拟合指标为字典格式
    fit_indices = {
        'Chi-Square': fit_obj.fmin,
        'CFI': fit_obj.cfi,
        'TLI': fit_obj.tli,
        'RMSEA': fit_obj.rmsea,
        'SRMR': fit_obj.srmr,
        'AIC': fit_obj.aic,
        'BIC': fit_obj.bic
    }

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # 参数估计表
        param_sheet = param_df.round(3).reset_index()
        param_sheet.to_excel(writer, sheet_name='参数估计', index=False)

        # 拟合指标表（转换为DataFrame）
        fit_sheet = pd.DataFrame(list(fit_indices.items()),
                                 columns=['指标', '值']).round(3)
        fit_sheet.to_excel(writer, sheet_name='模型拟合', index=False)

        # 标准化解表
        std_solution_sheet = std_solution.round(3).reset_index()
        std_solution_sheet.to_excel(writer, sheet_name='标准化解', index=False)

        # 获取工作簿对象进行格式设置
        workbook = writer.book

        # 设置统一格式
        format_header = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4F81BD',
            'font_color': 'white',
            'border': 1
        })

        # 设置数值格式
        num_format = workbook.add_format({'num_format': '0.000'})

        # 应用格式到各工作表
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]

            # 设置列宽自适应
            for col_num, value in enumerate(param_sheet.columns.values):
                max_len = max((
                    param_sheet[value].astype(str).map(len).max(),
                    len(str(value))
                )) + 2
                worksheet.set_column(col_num, col_num, max_len)

            # 添加首行格式
            worksheet.autofilter(0, 0, 0, len(param_sheet.columns) - 1)
            worksheet.freeze_panes(1, 0)

            # 应用数字格式
            if sheet_name == '参数估计':
                worksheet.set_column('D:F', None, num_format)
            elif sheet_name == '模型拟合':
                worksheet.set_column('B:B', None, num_format)

        # 添加解释说明页
        notes = [
            ("参数估计表说明:", [
                "Estimate: 非标准化路径系数",
                "Std. Err: 标准误",
                "z-value: Wald检验值",
                "p-value: ***p<0.001, **p<0.01, *p<0.05"
            ]),
            ("模型拟合指标说明:", [
                "CFI: 比较拟合指数 >0.90",
                "TLI: Tucker-Lewis指数 >0.90",
                "RMSEA: 近似误差均方根 <0.08",
                "SRMR: 标准化残差均方根 <0.08"
            ])
        ]

        notes_sheet = workbook.add_worksheet('结果解读')
        row = 0
        for title, items in notes:
            notes_sheet.write(row, 0, title, format_header)
            row += 1
            for item in items:
                notes_sheet.write(row, 0, item)
                row += 1
            row += 1  # 空行分隔

    print(f"\n分析结果已保存至: {filename}")


# ------------------------- 增强可视化模块 -------------------------
def enhanced_sem_visualization(param_df, fit_result):
    """专业级SEM路径图绘制"""
    plt.figure(figsize=(20, 15))
    G = nx.DiGraph()

    # 节点分类与属性设置
    latent_vars = list(set(param_df[param_df['op'] == '=~']['lval']))
    obs_vars = list(set(param_df[param_df['op'] == '=~']['rval']))

    # 添加节点（确保所有节点都有type属性）
    for lv in latent_vars:
        G.add_node(lv, subset='latent', type='latent', label=lv)
    for ov in obs_vars:
        G.add_node(ov, subset='observed', type='observed', label=ov)

    # 添加边关系
    for _, row in param_df.iterrows():
        if row['op'] == '=~':
            G.add_edge(row['lval'], row['rval'],
                       type='measurement',
                       label=f"λ={row['Estimate']:.2f}",
                       pvalue=row['p-value'])
        elif row['op'] == '~':
            G.add_edge(row['rval'], row['lval'],
                       type='structural',
                       label=f"β={row['Estimate']:.2f}\n(p={row['p-value']:.3f})")

        # 修正布局参数
        pos = nx.multipartite_layout(G, subset_key='subset',  # 使用subset属性
                                     align='horizontal',
                                     scale=2.5)

    # 调整节点位置
    for node, (x, y) in pos.items():
        if G.nodes[node]['type'] == 'latent':
            pos[node] = (x * 1.5, y + 0.2)
        else:
            pos[node] = (x, y - 0.3)

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=latent_vars,
        node_color='#4CAF50',
        node_size=4000,
        node_shape='s'
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=obs_vars,
        node_color='#2196F3',
        node_size=1500,
        node_shape='o'
    )

    # 绘制边
    measurement_edges = [(u, v) for u, v, d in G.edges(data=True)
                         if d['type'] == 'measurement']
    structural_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if d['type'] == 'structural']

    nx.draw_networkx_edges(
        G, pos,
        edgelist=measurement_edges,
        edge_color='#666666',
        style='dashed',
        width=2
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=structural_edges,
        edge_color='#E91E63',
        width=3,
        arrowsize=25,
        arrowstyle='->'
    )

    # 添加标签
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_family='SimHei'
    )

    # 边标签处理
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    for edge, label in edge_labels.items():
        if 'p=' in label:
            pval = float(label.split('p=')[1].strip(')'))
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            edge_labels[edge] += stars

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        label_pos=0.6,
        bbox=dict(alpha=0)
    )

    # 专业图例
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='潜变量',
                   markersize=15, markerfacecolor='#4CAF50'),
        plt.Line2D([0], [0], marker='o', color='w', label='观测变量',
                   markersize=10, markerfacecolor='#2196F3'),
        plt.Line2D([0], [0], color='#666666', linestyle='--', lw=2, label='因子载荷'),
        plt.Line2D([0], [0], color='#E91E63', lw=3, label='结构路径')
    ]

    plt.legend(handles=legend_elements, loc='upper right',
               fontsize=12, framealpha=0.9)

    plt.title("完整结构方程模型路径图\n(含测量模型与结构模型)",
              fontsize=18, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('Full_SEM_Path_Diagram.png', dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------- 在模型拟合后调用 -------------------------
enhanced_sem_visualization(param_df, fit_result)

# 调用导出函数时传递fit_result
save_sem_results(param_df, fit_result, std_solution, 'SEM分析结果.xlsx')
