import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy.stats import ttest_ind

try:
    from semopy import Model, calc_AVE_CR
except ImportError:
    print("警告: semopy库未安装，部分验证性因子分析功能将受限")

# 环境配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def initialize_environment():
    """初始化文件目录"""
    Path("visualizations").mkdir(exist_ok=True)
    Path("analysis_results").mkdir(exist_ok=True)


def simplify_column_name(name):
    """列名简化函数"""
    match = re.match(r'^(\d+)、.*?(\d+)\.', name)
    if match:
        return f"Q{match.group(1)}_{match.group(2)}"
    return name


def cronbach_alpha(data):
    """Cronbach's α计算"""
    data_clean = data.dropna(thresh=data.shape[1] * 0.7)
    n = data_clean.shape[1]
    if n < 2:
        return np.nan, len(data_clean)
    item_var = data_clean.var(axis=0, ddof=1)
    total_var = data_clean.sum(axis=1).var(ddof=1)
    alpha = (n / (n - 1)) * (1 - item_var.sum() / total_var)
    return alpha, len(data_clean)


def validate_factor_model(data, model_spec):
    """验证性因子分析"""
    model = Model(model_spec)
    model.fit(data)
    estimates = model.inspect(std_est=True)
    ave, cr = calc_AVE_CR(model)
    fit = model.inspect()
    return {
        'fit_indices': {
            'CFI': fit['CFI'],
            'TLI': fit['TLI'],
            'RMSEA': fit['RMSEA'],
            'SRMR': fit['SRMR']
        },
        'standardized_loadings': estimates[['lval', 'std_est']],
        'AVE': ave,
        'CR': cr
    }


def analyze_demographics(df):
    """人口统计学分析"""
    demo_results = {}

    # 性别分布
    gender = df['性别'].value_counts(normalize=True).mul(100).round(2)
    demo_results['性别'] = pd.DataFrame({
        '类别': gender.index,
        '百分比(%)': gender.values
    })

    # 专业背景分布
    major = df['专业背景'].value_counts(normalize=True).mul(100).round(2)
    demo_results['专业背景'] = pd.DataFrame({
        '类别': major.index,
        '百分比(%)': major.values
    })

    # 年龄分布（示例数据需要实际年龄字段）
    if '年龄' in df:
        age_groups = pd.cut(df['年龄'],
                            bins=[0, 22, 25, 30, 100],
                            labels=['≤22', '22-25', '26-30', '≥30'])
        age_dist = age_groups.value_counts(normalize=True).mul(100).round(2)
        demo_results['年龄'] = pd.DataFrame({
            '年龄段': age_dist.index,
            '百分比(%)': age_dist.values
        })

    return demo_results


def analyze_core_variables(df):
    """核心变量分析"""
    results = {}

    # 计算量表总分
    df['AI能力'] = df.filter(regex='AI能力').mean(axis=1)
    df['学习适应'] = df.filter(regex='学习适应').mean(axis=1)
    df['科研效能'] = df.filter(regex='科研效能').mean(axis=1)

    # 描述统计
    desc = df[['AI能力', '学习适应', '科研效能']].describe().T
    desc = desc[['mean', 'std', 'min', 'max']]
    desc.columns = ['均值', '标准差', '最小值', '最大值']
    results['描述统计'] = desc.round(3)

    # 群体差异（示例：性别差异）
    male = df[df['性别'] == '男']
    female = df[df['性别'] == '女']
    t_test = pd.DataFrame()
    for var in ['AI能力', '学习适应', '科研效能']:
        t_stat, p_val = ttest_ind(male[var], female[var], nan_policy='omit')
        t_test[var] = [t_stat, p_val]
    t_test.index = ['t值', 'p值']
    results['性别差异'] = t_test.round(3)

    return results


def analyze_scale(data, section_name, n_factors):
    """量表综合分析"""
    data.columns = [simplify_column_name(col) for col in data.columns]
    data_clean = data.dropna()

    with pd.ExcelWriter(f"analysis_results/{section_name}_分析报告.xlsx") as writer:
        # 信度分析
        alpha, n = cronbach_alpha(data)
        pd.DataFrame({'Cronbach α': [alpha], '样本量': [n]}).to_excel(writer, sheet_name="信度", index=False)

        # 验证性因子分析
        try:
            model_spec = f'''
            # 根据实际量表结构调整模型
            factor1 =~ {' + '.join(data.columns[:3])}
            factor2 =~ {' + '.join(data.columns[3:6])}
            '''
            cfa = validate_factor_model(data_clean, model_spec)

            # 保存结果
            pd.DataFrame([cfa['fit_indices']]).to_excel(writer, sheet_name="模型拟合")
            cfa['standardized_loadings'].to_excel(writer, sheet_name="因子载荷")
            pd.DataFrame({
                '潜变量': list(cfa['AVE'].keys()),
                'AVE': [round(v, 3) for v in cfa['AVE'].values()],
                'CR': [round(v, 3) for v in cfa['CR'].values()]
            }).to_excel(writer, sheet_name="效度指标", index=False)
        except Exception as e:
            print(f"验证性因子分析失败: {str(e)}")

        # 共同方法偏差
        try:
            fa = FactorAnalyzer(n_factors=1, rotation=None)
            fa.fit(data_clean)
            variance = fa.get_factor_variance()[0][0]
            pd.DataFrame({
                '首因子解释率(%)': [variance * 100],
                '判定结果': ['不显著' if variance < 0.4 else '显著']
            }).to_excel(writer, sheet_name="共同方法偏差", index=False)
        except Exception as e:
            print(f"共同方法偏差检验失败: {str(e)}")


if __name__ == "__main__":
    initialize_environment()

    try:
        df = pd.read_excel("研究生AI能力、因素、学习适应、科研效能感.xlsx")

        # 人口统计学分析
        demo_results = analyze_demographics(df)
        with pd.ExcelWriter("analysis_results/人口统计学.xlsx") as writer:
            for sheet, data in demo_results.items():
                data.to_excel(writer, sheet_name=sheet, index=False)

        # 核心变量分析
        core_results = analyze_core_variables(df)
        with pd.ExcelWriter("analysis_results/核心变量.xlsx") as writer:
            for sheet, data in core_results.items():
                data.to_excel(writer, sheet_name=sheet)

        # 量表分析
        sections = [
            ('AI能力', df.filter(regex='AI能力'), 4),
            ('学习适应', df.filter(regex='学习适应'), 2),
            ('科研效能', df.filter(regex='科研效能'), 2)
        ]
        for name, data, factors in sections:
            analyze_scale(data, name, factors)

        print("分析完成！结果保存在analysis_results目录")

    except Exception as e:
        print(f"分析失败: {str(e)}")
