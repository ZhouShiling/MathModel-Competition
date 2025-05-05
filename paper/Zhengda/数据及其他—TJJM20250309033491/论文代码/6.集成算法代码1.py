import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score, median_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# 全局样式配置
# plt.style.use('seaborn-talk')
sns.set_palette("husl")
plt.rcParams.update({
    'font.sans-serif': 'Microsoft YaHei',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.facecolor': 'white'
})


class ModelAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)
        self.feature_names = [f'Q18_{i}' for i in range(1, 6)] + [f'Q19_{i}' for i in range(1, 11)]
        self.X, self.y, self.X_test, self.y_test = self._prepare_data()
        self.models = self._init_models()
        self.results = []

    def _prepare_data(self):
        """数据预处理"""
        X = self.df[self.feature_names]
        y = self.df['AI能力总分']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        return X_train, y_train, X_test, y_test

    def _init_models(self):
        """初始化模型集合"""
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=0.5),
            "Lasso": Lasso(alpha=0.01),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "DecisionTree": DecisionTreeRegressor(max_depth=4),
            "RandomForest": RandomForestRegressor(n_estimators=100),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', max_depth=3),
            "LightGBM": lgb.LGBMRegressor(num_leaves=31),
            "SVR": SVR(kernel='rbf', C=100),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
            "MLP": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
        }

    def _calc_metrics(self, y_true, y_pred, n_features):
        """计算完整评估指标"""
        return {
            "R²": r2_score(y_true, y_pred),
            "Adj R²": 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - n_features - 1),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "EVS": explained_variance_score(y_true, y_pred),
            "Max Error": np.max(np.abs(y_true - y_pred)),
            "MedAE": median_absolute_error(y_true, y_pred)
        }

    def _get_importance(self, model):
        """获取特征重要性"""
        if hasattr(model, 'coef_'):
            return model.coef_
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return np.zeros(self.X.shape[1])

    def _save_plot_data(self, data_dict, model_name, plot_type):
        """保存绘图数据到CSV"""
        df = pd.DataFrame(data_dict)
        df.to_csv(f"{model_name}_{plot_type}_data.csv",
                  index=False,
                  encoding='utf_8_sig',
                  float_format="%.4f")

    def _plot_feature_importance(self, importance, model_name):
        """特征重要性可视化"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance, y=self.feature_names, ax=ax)
        ax.set_title(f"{model_name} - 特征重要性", fontsize=14)
        ax.set_xlabel("重要性得分")
        plt.tight_layout()
        plt.savefig(f"{model_name}_feature_importance.png")
        plt.close()

        # 保存特征重要性数据
        self._save_plot_data({
            "Feature": self.feature_names,
            "Importance": importance
        }, model_name, "feature_importance")

    def _plot_prediction(self, y_true, y_pred, metrics, model_name):
        """预测效果可视化"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x=y_true, y=y_pred,
                    scatter_kws={'alpha': 0.6, 'color': '#2980b9'},
                    line_kws={'color': '#e67e22', 'lw': 2},
                    ax=ax)

        # 添加指标标注
        textstr = '\n'.join((
            f'R² = {metrics["R²"]:.3f}',
            f'Adj R² = {metrics["Adj R²"]:.3f}',
            f'RMSE = {metrics["RMSE"]:.2f}',
            f'MAE = {metrics["MAE"]:.2f}'))

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()], '--', color='#7f8c8d')
        ax.set_title(f"{model_name}预测效果", fontsize=14)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        plt.tight_layout()
        plt.savefig(f"{model_name}_prediction.png")
        plt.close()

        # 保存预测数据
        self._save_plot_data({
            "True_Value": y_true.values,
            "Predicted_Value": y_pred
        }, model_name, "prediction")

    def _plot_residuals(self, residuals, metrics, model_name):
        """残差分析可视化"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color='#9b59b6', bins=20, ax=ax)
        ax.axvline(0, color='#c0392b', linestyle='--', linewidth=1.5)
        ax.set_title(f"{model_name}残差分布\nMSE={metrics['MSE']:.2f}, MedAE={metrics['MedAE']:.2f}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{model_name}_residuals.png")
        plt.close()

        # 保存残差数据
        self._save_plot_data({
            "Residual": residuals,
            "Absolute_Error": np.abs(residuals)
        }, model_name, "residuals")

    def _shap_analysis(self, model, X, model_name):
        """SHAP特征影响分析"""
        try:
            if 'tree' in str(type(model)):
                explainer = shap.TreeExplainer(model)
            elif 'linear' in str(type(model)):
                explainer = shap.LinearExplainer(model, X)
            else:
                explainer = shap.KernelExplainer(model.predict, X)

            shap_values = explainer.shap_values(X)

            # 可视化
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            plt.title(f"{model_name} - SHAP特征影响", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{model_name}_shap.png")
            plt.close()

            # 保存SHAP值数据
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            pd.DataFrame(shap_values, columns=self.feature_names) \
                .to_csv(f"{model_name}_shap_values.csv",
                        index=False,
                        encoding='utf_8_sig',
                        float_format="%.4f")

        except Exception as e:
            print(f"SHAP分析失败 ({model_name}): {str(e)}")

    def run_analysis(self):
        """执行完整分析流程"""
        for name, model in self.models.items():
            print(f"\n正在分析: {name}")

            # 训练模型
            model.fit(self.X, self.y)

            # 预测与评估
            y_pred = model.predict(self.X_test)
            metrics = self._calc_metrics(self.y_test, y_pred, self.X.shape[1])

            # 特征重要性
            importance = self._get_importance(model)

            # 生成可视化
            self._plot_feature_importance(importance, name)
            self._plot_prediction(self.y_test, y_pred, metrics, name)
            self._plot_residuals(self.y_test - y_pred, metrics, name)
            self._shap_analysis(model, self.X_test, name)

            # 保存结果
            self.results.append({
                "Model": name,
                **metrics
            })

        # 保存评估结果
        result_df = pd.DataFrame(self.results)
        result_df = result_df.sort_values("RMSE", ascending=True).round(3)

        # 优化Excel输出
        with pd.ExcelWriter("model_performance.xlsx", engine='xlsxwriter') as writer:
            result_df.to_excel(writer, index=False, sheet_name='模型性能')

            # 设置自动列宽
            workbook = writer.book
            worksheet = writer.sheets['模型性能']
            for idx, col in enumerate(result_df.columns):
                series = result_df[col]
                max_len = max((
                    series.astype(str).map(len).max(),
                    len(str(series.name))
                )) + 2
                worksheet.set_column(idx, idx, max_len)

        # 生成模型对比图
        plt.figure(figsize=(10, 6))
        sns.barplot(x="R²", y="Model", data=result_df, palette="Blues_d")
        plt.title("回归模型性能对比 (按R²排序)", fontsize=14)
        plt.xlabel("决定系数 R²", fontsize=12)
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.close()

        # 保存元数据
        pd.DataFrame.from_dict({
            "feature_names": [",".join(self.feature_names)],
            "train_samples": [len(self.y)],
            "test_samples": [len(self.y_test)],
            "analysis_date": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]
        }).to_csv("analysis_metadata.csv", index=False, encoding='utf_8_sig')


if __name__ == "__main__":
    analyzer = ModelAnalyzer("研究生AI能力分析结果.xlsx")
    analyzer.run_analysis()
