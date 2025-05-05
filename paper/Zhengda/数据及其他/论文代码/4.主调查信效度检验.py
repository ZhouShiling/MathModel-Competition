from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from scipy.stats import pearsonr, shapiro, kstest
from sklearn.preprocessing import StandardScaler

# 环境配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def initialize_environment():
    """初始化文件目录"""
    Path("visualizations").mkdir(exist_ok=True)
    Path("analysis_results").mkdir(exist_ok=True)


def cronbach_alpha(data):
    """改进的Cronbach's α计算"""
    data_clean = data.dropna(thresh=data.shape[1] * 0.7)
    n = data_clean.shape[1]
    if n < 2:
        return np.nan, len(data_clean)
    item_var = data_clean.var(axis=0, ddof=1)
    total_var = data_clean.sum(axis=1).var(ddof=1)
    alpha = (n / (n - 1)) * (1 - item_var.sum() / total_var)
    return alpha, len(data_clean)


def factor_reliability_analysis(data, loadings, n_factors):
    """因子维度信度分析"""
    factor_items = {f"因子{i + 1}": [] for i in range(n_factors)}
    for idx, row in loadings.iterrows():
        max_factor = row.abs().idxmax()
        factor_items[max_factor].append(idx)
    results = []
    for factor, items in factor_items.items():
        if len(items) >= 2:
            subset = data[items]
            alpha, n = cronbach_alpha(subset)
            results.append({
                '因子名称': factor,
                '题项数量': len(items),
                'Cronbach\'s α': alpha,
                '有效样本量': n,
                '包含题项': '; '.join(items)
            })
        else:
            results.append({
                '因子名称': factor,
                '题项数量': len(items),
                'Cronbach\'s α': np.nan,
                '有效样本量': np.nan,
                '包含题项': '; '.join(items) if items else '无'
            })
    return pd.DataFrame(results)


def enhanced_item_analysis(data, section_name):
    """单变量增强分析"""
    results = []
    for col in data.columns:
        item_data = data[[col]].dropna()
        if len(item_data) < 10:
            continue
        mean = item_data[col].mean()
        std = item_data[col].std()
        ks_stat, ks_p = kstest(item_data[col], 'norm', args=(mean, std))
        shapiro_stat, shapiro_p = shapiro(item_data[col]) if 3 < len(item_data) < 5000 else (np.nan, np.nan)
        total_scores = data.mean(axis=1)
        corr_total, _ = pearsonr(item_data[col], total_scores)
        corr_others = data.corrwith(item_data[col]).mean()
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            fa = FactorAnalyzer(n_factors=1, rotation=None)
            fa.fit(scaled_data)
            factor_loading = fa.loadings_[data.columns.get_loc(col)][0]
        except:
            factor_loading = np.nan
        results.append({
            '变量名': col,
            '样本量': len(item_data),
            '均值': mean,
            '标准差': std,
            '偏度': item_data[col].skew(),
            '峰度': item_data[col].kurtosis(),
            'KS统计量': ks_stat,
            'KS_p值': ks_p,
            'Shapiro统计量': shapiro_stat,
            'Shapiro_p值': shapiro_p,
            '题目区分度': corr_total,
            '平均题目相关': corr_others,
            '因子载荷': factor_loading
        })
        save_item_visualization(item_data[col], section_name, col)
    return pd.DataFrame(results)


def save_item_visualization(data, section_name, variable_name):
    """保存单变量可视化"""
    save_dir = Path(f"visualizations/{section_name}")
    save_dir.mkdir(exist_ok=True)
    try:
        plt.figure(figsize=(10, 5))
        sns.histplot(data, kde=True, color='#2c7fb8')
        plt.title(f'{variable_name} 分布')
        plt.savefig(save_dir / f"{variable_name}_分布.png")
        plt.close()
        plt.figure(figsize=(10, 5))
        sm.qqplot(data, line='s', marker='o', markerfacecolor='#2c7fb8', markeredgecolor='#2c7fb8')
        plt.title(f'{variable_name} Q-Q图')
        plt.savefig(save_dir / f"{variable_name}_Q-Q图.png")
        plt.close()
    except Exception as e:
        print(f"可视化保存失败: {str(e)}")


def analyze_scale(data, section_name, n_factors):
    """整体量表分析（含相关性分析）"""
    save_dir = Path(f"visualizations/{section_name}")
    save_dir.mkdir(exist_ok=True)

    with pd.ExcelWriter(f"analysis_results/{section_name}_分析报告.xlsx") as writer:
        # 整体信度
        alpha, valid_n = cronbach_alpha(data)
        pd.DataFrame({'Cronbach\'s α': [alpha], '有效样本量': [valid_n]}).to_excel(writer, sheet_name="整体信度",
                                                                                   index=False)

        # 整体效度
        try:
            kmo_all, kmo_model = calculate_kmo(data)
            bartlett_stat, bartlett_p = calculate_bartlett_sphericity(data)
            pd.DataFrame({
                'KMO值': [kmo_model],
                'Bartlett检验统计量': [bartlett_stat],
                'Bartlett_p值': [bartlett_p]
            }).to_excel(writer, sheet_name="整体效度", index=False)
        except Exception as e:
            print(f"效度分析失败: {str(e)}")

        # 变量相关性分析
        try:
            data_clean = data.dropna()
            if len(data_clean) > 1:
                corr_matrix = data_clean.corr()
                corr_matrix.to_excel(writer, sheet_name="变量相关系数矩阵")

                plt.figure(figsize=(15, 12))
                sns.heatmap(corr_matrix, annot=False, cmap="RdBu_r", center=0,
                            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
                plt.title(f"{section_name}变量间相关系数矩阵", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(save_dir / f"{section_name}_变量相关热力图.png", dpi=300)
                plt.close()
        except Exception as e:
            print(f"相关性分析异常: {str(e)}")

        # 单变量分析
        item_df = enhanced_item_analysis(data, section_name)
        item_df.to_excel(writer, sheet_name="单变量分析", index=False)

        # 因子分析
        try:
            scaled_data = StandardScaler().fit_transform(data.dropna())
            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
            fa.fit(scaled_data)
            variance, prop_var, cum_var = fa.get_factor_variance()
            loadings = pd.DataFrame(fa.loadings_, index=data.columns,
                                    columns=[f"因子{i + 1}" for i in range(n_factors)])
            factor_alpha_df = factor_reliability_analysis(data, loadings, n_factors)
            loadings.to_excel(writer, sheet_name="因子载荷")
            pd.DataFrame({
                '因子': [f'因子{i + 1}' for i in range(n_factors)] + ['累计'],
                '特征根': list(variance) + [np.sum(variance)],
                '方差解释率(%)': list(prop_var * 100) + [np.sum(prop_var) * 100],
                '累积方差解释率(%)': list(cum_var * 100) + [np.nan]
            }).to_excel(writer, sheet_name="方差解释率", index=False)
            factor_alpha_df.to_excel(writer, sheet_name="因子信度", index=False)
        except Exception as e:
            print(f"因子分析失败: {str(e)}")


if __name__ == "__main__":
    initialize_environment()

    try:
        df = pd.read_excel("研究生AI能力、因素、学习适应、科研效能感.xlsx")
        total_scores = pd.DataFrame()

        sections = [
            ('AI能力评估',
             df.loc[:, '9、人本思维能力—1.在实践中，我能够主动选择并使用数字工具':'12、12.我能够利用AI技术实现知识的转化'],
             4),
            ('AI能力影响因素', df.loc[:,
                               '14、个体认知—1.我能够利用AI技术解决在学习中出现的问题':'17、18.我认为使用AI技术可以更容易达到学习或科研目标'],
             4),
            ('学习适应量表',
             df.loc[:, '18、学习适应量表—1.我感觉我适应研究生的学习':'19、10.因为学业，放弃课外活动让我感到失望。'], 2),
            ('科研效能感量表',
             df.loc[:, '20、科研效能感量表—1.遵循研究伦理':'20、9.发现并报告研究的局限性和未来可能的研究方向'], 2)
        ]

        # 各量表分析
        for name, data, factors in sections:
            print(f"\n{'=' * 30}\n正在分析: {name}\n{'=' * 30}")
            analyze_scale(data, name, factors)
            total_scores[name] = data.mean(axis=1)

        # 跨量表总分分析
        if not total_scores.empty:
            with pd.ExcelWriter("analysis_results/量表总分相关分析.xlsx") as writer:
                score_corr = total_scores.corr()
                score_corr.to_excel(writer, sheet_name="总分相关系数")

                plt.figure(figsize=(10, 8))
                sns.heatmap(score_corr, annot=True, fmt=".2f", cmap="Blues",
                            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
                plt.title("量表总分间相关系数矩阵", fontsize=14)
                plt.tight_layout()
                plt.savefig("visualizations/量表总分相关热力图.png", dpi=300)
                plt.close()

                total_scores.describe().to_excel(writer, sheet_name="总分描述统计")

        print("\n分析完成！结果目录：")
        print("- 统计报告: analysis_results/")
        print("- 可视化图表: visualizations/")

    except Exception as e:
        print(f"分析终止: {str(e)}")
    finally:
        plt.close('all')
