import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 假设数据已经保存在一个CSV文件中，这里我们加载数据
data = pd.read_excel('000858历史日数据.xlsx')

# 提取特征和标签
X = data[['收盘', '开盘', '高', '低', '交易量']]
y = data['涨跌情况（1为涨、0为跌）']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练判别分析模型
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 预测测试集的结果
predictions = lda.predict(X_test)
print(predictions)
# 输出分类报告
print(classification_report(y_test, predictions))

# 输出混淆矩阵
cm = confusion_matrix(y_test, predictions)
print(cm)

# 计算AUC值
auc = roc_auc_score(y_test, predictions)
print("AUC Score:", auc)

# 输出特征值
print("Explained Variance Ratios (Feature Values):")
print(lda.explained_variance_ratio_)

# 输出Wilks' Lambda
print("Wilks' Lambda:")
wilks_lambda = 1 - lda.explained_variance_ratio_[0]
if wilks_lambda == 1:
    wilks_lambda = 0.999999  # 微小调整，避免除零
print(wilks_lambda)

# 输出卡方统计量和自由度
df = len(lda.scalings_)
chi_square_statistic = -2 * np.log(wilks_lambda)
p_value = np.exp(-chi_square_statistic / df)
print("Chi-square statistic:", chi_square_statistic)
print("Degrees of Freedom:", df)
print("P-value:", p_value)

# 输出标准化典型系数
print("Standardized Canonical Coefficients:")
print(lda.scalings_)

# 输出未标准化典型系数
print("Unstandardized Canonical Coefficients:")
print(lda.coef_)

# 输出典型结构矩阵
print("Canonical Structure Matrix:")
print(lda.coef_ / np.sqrt(np.sum(lda.coef_ ** 2, axis=1))[:, np.newaxis])

# 输出典型组平均值
print("Group Means on Canonical Variables:")
group_means = np.array([np.mean(lda.transform(X_train)[y_train == 1]),
                        np.mean(lda.transform(X_train)[y_train == 0])])
print(group_means)

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['跌', '涨'], yticklabels=['跌', '涨'])
plt.title('判别分析的混淆矩阵')
plt.ylabel('实际')
plt.xlabel('预测')
plt.show()

# 可视化LDA投影后的数据点
lda_transformed = lda.transform(X_train)
plt.figure(figsize=(8, 6))
colors = ['red', 'blue']
for color, i, target_name in zip(colors, [1, 0], ['涨', '跌']):
    plt.scatter(lda_transformed[y_train == i, 0], np.zeros((lda_transformed[y_train == i].shape[0],)),
                color=color, lw=2, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('000858数据集的LDA')
plt.xlabel('LDA组件')
plt.show()

# 使用交叉验证评估模型的稳定性
cv_scores = cross_val_score(lda, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {np.mean(cv_scores)}")

# 使用交叉验证评估AUC
cv_auc_scores = cross_val_score(lda, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC scores: {cv_auc_scores}")
print(f"Mean cross-validation AUC score: {np.mean(cv_auc_scores)}")

# 绘制ROC曲线
y_pred_prob = lda.predict_proba(X_test)[:, 1]  # 获取正类的概率
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='o')
plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
