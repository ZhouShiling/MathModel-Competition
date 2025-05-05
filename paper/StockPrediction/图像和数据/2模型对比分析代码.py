import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 载入数据
data = pd.read_excel('000858历史日数据.xlsx')

# 指定自变量和因变量
X = data[['收盘', '开盘', '高', '低', '交易量']]
y = data['涨跌情况（1为涨、0为跌）']

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 训练逻辑回归模型
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# 输出回归系数
coef = logistic_model.coef_
intercept = logistic_model.intercept_
print(f"Coefficients: {coef}, Intercept: {intercept}")

# 预测测试集
y_pred = logistic_model.predict(X_test)
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]

# 计算混淆矩阵和分类报告
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# 计算AUC值
auc = roc_auc_score(y_test, y_pred_prob)
print("AUC Score:", auc)

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['跌', '涨'], yticklabels=['跌', '涨'])
plt.title('Logistic回归的混淆矩阵')
plt.ylabel('实际')
plt.xlabel('预测')
plt.show()

# 绘制ROC曲线
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

# 使用交叉验证评估模型的稳定性
cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {np.mean(cv_scores)}")

# 使用交叉验证评估AUC
cv_auc_scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC scores: {cv_auc_scores}")
print(f"Mean cross-validation AUC score: {np.mean(cv_auc_scores)}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['跌', '涨'], yticklabels=['跌', '涨'])
plt.title('混淆矩阵')
plt.ylabel('实际')
plt.xlabel('预测')
plt.show()
