import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# 在文件开头设置全局样式
# plt.style.use('seaborn-darkgrid')
color_palette = sns.color_palette("husl", 8)
sns.set_palette(color_palette)


# 数据预处理
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # 将 text 转换为字符串类型
    text = str(text)
    # 统一同义词替换
    replacements = {
        'gpt': 'ChatGPT', '文心': '文心一言', 'kimi': 'KIMI',
        '伦理': 'AI伦理', '隐私': '数据隐私', '抄袭': '学术不端'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


# 加载数据
df = pd.read_excel('研究生AI能力、因素、学习适应、科研效能感1.xlsx')

# 配置分析参数
questions = {
    'q1': '13、开放式问题：请用短词或者短句回答—1.你了解哪些AI模型，例如？',
    'q2': '13、2.在人工智能应用中，你最感兴趣的领域是什么？',
    'q3': '13、3.AI伦理问题研究涉及哪些方面？',
    'q5': '13、5.当你遇到一个新的AI技术时，你通常如何进行学习和掌握？',
    'q6': '13、6.你认为AI技术的哪些发展领域可能在未来取得突破？',
    'q7': '13、7.AI项目通常需要团队协作，你如何在团队中发挥作用？',
    'q8': '13、8.你认为未来几年内AI技术会有哪些重要的突破或创新？',
}


# 1. AI模型认知分析
def analyze_ai_models(series):
    # 高频模型提取
    texts = series.apply(preprocess_text).str.cat(sep=' ')
    tags = jieba.analyse.extract_tags(texts, topK=20, withWeight=True)

    # 分类统计
    categories = {
        'NLP': ['ChatGPT', '文心一言', '通义千问'],
        '多模态': ['KIMI', '智谱清言'],
        '教育专用': ['讯飞星火', '豆包']
    }

    category_count = {k: 0 for k in categories}
    for tag, weight in tags:
        for cat, models in categories.items():
            if tag in models:
                category_count[cat] += weight

    # 可视化
    plt.figure(figsize=(10, 6))
    pd.Series(category_count).plot(kind='barh')
    plt.title('AI模型认知领域分布')
    plt.savefig('model_category.jpg', dpi=300, bbox_inches='tight')
    # 增强可视化
    plt.figure(figsize=(12, 8), dpi=120)
    ax = pd.Series(category_count).sort_values().plot(
        kind='barh',
        color=['#4C72B0', '#55A868', '#C44E52'],  # 学术风格配色
        edgecolor='w',
        linewidth=2
    )

    # 添加数据标签
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.1, p.get_y() + 0.2,
                f'{width:.1f}',
                ha='center', va='center',
                fontsize=10)

    plt.title('AI模型认知领域分布', fontsize=14, pad=20)
    plt.xlabel('认知权重', labelpad=10)
    plt.ylabel('技术领域', labelpad=10)
    plt.tight_layout()
    plt.savefig('model_category.jpg', bbox_inches='tight', dpi=300)
    plt.close()

    return pd.DataFrame(tags, columns=['模型', '权重'])


# 2. 兴趣领域主题建模
def topic_modeling(series, n_topics=3):
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2)
    dtm = tfidf.fit_transform(series.dropna())

    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(dtm)

    # 可视化主题词
    features = tfidf.get_feature_names_out()
    plt.figure(figsize=(12, 8))
    for i, topic in enumerate(lda.components_):
        top_words = [features[j] for j in topic.argsort()[:-10:-1]]
        plt.subplot(1, n_topics, i + 1)
        plt.barh(top_words, topic[topic.argsort()[:-10:-1]])
        plt.title(f'Topic {i + 1}')
    plt.tight_layout()
    plt.savefig('interest_topics.jpg', dpi=300)
    # 增强可视化
    plt.figure(figsize=(15, 8), dpi=120)
    colors = ['#2E86C1', '#17A589', '#D4AC0D'][:n_topics]

    for i, (topic, color) in enumerate(zip(lda.components_, colors)):
        top_words = [features[j] for j in topic.argsort()[:-8:-1]]
        values = topic[topic.argsort()[:-8:-1]]

        plt.subplot(1, n_topics, i + 1)
        sns.barplot(x=values, y=top_words,
                    palette=sns.light_palette(color, n_colors=8),
                    edgecolor='.2')

        plt.title(f'主题{i + 1}', fontsize=12, pad=10)
        plt.xlabel('特征权重', fontsize=10)
        plt.tick_params(axis='both', labelsize=9)

    plt.suptitle('兴趣领域主题分布', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('interest_topics.jpg', dpi=300)
    plt.close()


# 3. 伦理问题语义网络
def build_semantic_network(series):
    # 计算 ethics_score
    ethics_keywords = ['隐私', '安全', '版权']
    ethics_score = series.apply(
        lambda x: sum(1 for w in jieba.lcut(str(x)) if w in ethics_keywords)
    )

    # 构建语义网络
    texts = series.apply(preprocess_text).tolist()
    all_words = []
    for text in texts:
        words = [w for w in jieba.lcut(text) if len(w) > 1]  # 过滤单字
        all_words.extend(words)

    # 构建共现矩阵
    unique_words = list(set(all_words))
    cooc = pd.DataFrame(0, index=unique_words, columns=unique_words)

    for text in texts:
        words = [w for w in jieba.lcut(text) if len(w) > 1]
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                if w1 in cooc.index and w2 in cooc.columns:
                    cooc.loc[w1, w2] += 1

    # 保存共现矩阵到Excel
    cooc.to_excel('semantic_cooccurrence.xlsx', sheet_name='共现矩阵')

    # 网络可视化（仅显示高频连接）
    G = nx.Graph()
    threshold = 2  # 共现次数阈值
    for w1 in cooc.index:
        for w2 in cooc.columns:
            if cooc.loc[w1, w2] > threshold:
                G.add_edge(w1, w2, weight=cooc.loc[w1, w2])

    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_size=[v * 100 for v in dict(G.degree).values()],
            width=[d['weight'] * 0.5 for u, v, d in G.edges(data=True)],
            font_size=10)
    plt.title("伦理问题语义网络（共现次数>2）")
    plt.savefig('ethics_network.jpg', dpi=300)
    plt.close()

    # 交叉分析箱线图
    plt.figure(figsize=(10, 6), dpi=120)
    sns.boxplot(
        x=df['8、您使用过哪些大语言模型【多选题】(ChatGPT)'].fillna(0),
        y=ethics_score,  # 使用函数内部计算的 ethics_score
        palette=sns.color_palette("Blues", 3),
        width=0.6,
        linewidth=2.5
    )

    # 添加数据点分布
    sns.stripplot(
        x=df['8、您使用过哪些大语言模型【多选题】(ChatGPT)'].fillna(0),
        y=ethics_score,  # 使用函数内部计算的 ethics_score
        color='#333333',
        alpha=0.4,
        size=6,
        jitter=0.2
    )

    plt.title("ChatGPT使用与伦理意识关系", fontsize=14, pad=15)
    plt.xlabel("是否使用ChatGPT", fontsize=12, labelpad=10)
    plt.ylabel("伦理关键词频次", fontsize=12, labelpad=10)
    plt.xticks([0, 1], ['未使用', '使用'], fontsize=10)
    plt.savefig('cross_analysis.jpg', dpi=300, bbox_inches='tight')
    plt.close()


from wordcloud import WordCloud
from collections import Counter


def generate_wordcloud(series, question_id, max_words=100, writer=None):
    # 加载停用词表（需准备停用词文件）
    with open('stopwords.txt', encoding='gbk') as f:
        stopwords = set(f.read().split())

    # 文本预处理与分词
    processed_text = series.apply(preprocess_text).str.cat(sep=' ')
    words = [w for w in jieba.lcut(processed_text)
             if len(w) > 1 and w not in stopwords]

    # 生成词云
    wc = WordCloud(
        font_path='msyh.ttc',  # 中文字体路径
        width=1600,
        height=1200,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
        # regexp=r"[\u4e00-\u9fa5]+"  # 仅保留中文
    ).generate(' '.join(words))

    # 可视化存储
    plt.figure(figsize=(20, 15))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'"{questions[question_id]}" 词云分析', fontsize=24, pad=20)
    plt.savefig(f'wordcloud_{question_id}.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    # 新增词频统计
    word_counts = Counter(words)
    freq_df = pd.DataFrame(word_counts.items(), columns=['词语', '频次']).sort_values('频次', ascending=False)

    # 保存到Excel
    if writer is not None:
        sheet_name = f"词频_{question_id}"
        freq_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # 限制sheet名称长度


# 执行分析
if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 创建Excel写入对象
    writer = pd.ExcelWriter('analysis_results.xlsx', engine='xlsxwriter')
    # 将问题列转换为字符串类型
    for qid in questions:
        df[questions[qid]] = df[questions[qid]].astype(str)
    # 问题1分析并保存
    model_df = analyze_ai_models(df[questions['q1']])
    model_df.to_excel(writer, sheet_name='AI模型认知')

    # 问题2分析
    topic_modeling(df[questions['q2']])

    # 问题3分析
    build_semantic_network(df[questions['q3']])

    # 交叉分析示例
    ethics_keywords = ['隐私', '安全', '版权']
    df['ethics_score'] = df[questions['q3']].apply(
        lambda x: sum(1 for w in jieba.lcut(str(x)) if w in ethics_keywords))

    plt.figure()
    sns.boxplot(x=df['8、您使用过哪些大语言模型【多选题】(ChatGPT)'].fillna(0),
                y=df['ethics_score'])
    plt.savefig('cross_analysis.jpg', dpi=300)
    # 保存交叉分析数据
    df[['ethics_score'] + list(questions.values())].to_excel(writer, sheet_name='原始数据')
    # 创建Excel写入对象（移动到主程序起始位置）
    writer = pd.ExcelWriter('词频统计结果.xlsx', engine='xlsxwriter')

    # 对每个问题生成词云并保存词频
    for qid in questions:
        generate_wordcloud(
            series=df[questions[qid]],
            question_id=qid,
            max_words=100,
            writer=writer  # 传递写入器对象
        )

    # 关闭Excel写入器
    writer.close()
