

# 安装必要库--> pip install -r requirements.txt
from openai import OpenAI  # 导入OpenAI客户端库
from sklearn.metrics import accuracy_score, classification_report  # 导入评估指标
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯分类器
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF特征提取
from tqdm import tqdm  # 用于显示进度条
from sklearn.model_selection import train_test_split  # 数据集分割
from sklearn.datasets import fetch_20newsgroups  # 20Newsgroups数据集
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
import re  # 正则表达式库

# 选择部分类别加快实验速度（可扩展）
categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.mideast']

# 数据加载和统计部分
print("\n" + "="*60)
print("数据集加载和统计分析".center(40))
print("="*60)

# 加载数据（带进度条）
with tqdm(total=1, desc="加载数据集") as pbar:
    newsgroups = fetch_20newsgroups(
        subset='all',  # 使用全部数据
        categories=categories,  # 指定类别
        remove=('headers', 'footers', 'quotes')  # 移除邮件头尾和引用
    )
    pbar.update(1)

# 打印数据集基本信息
print(f"数据集大小: {len(newsgroups.data)}")
print("\n[基本统计]")
print(f"总样本数: {len(newsgroups.data):,}")  # 千位分隔符
print("类别分布:")
unique_values, counts = np.unique(newsgroups.target, return_counts=True)
for idx, (value, count) in enumerate(zip(unique_values, counts)):
    print(
        f"{idx}. {newsgroups.target_names[value]:<25} {count:>4}个样本 ({count/len(newsgroups.data):.1%})")

# 数据分割统计
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target,
    test_size=0.2,  # 测试集占比20%
    random_state=42,  # 随机种子
    stratify=newsgroups.target  # 分层抽样保持类别比例
)

print("\n[数据分割]")
split_info = [
    ("训练集", X_train, y_train),
    ("测试集", X_test, y_test)
]

# 打印训练集和测试集分布
for name, data, labels in split_info:
    print(f"\n{name} ({len(data)}个样本, {len(data)/len(newsgroups.data):.1%}):")
    for idx in np.unique(labels):
        count = sum(labels == idx)
        print(
            f"  {idx}. {newsgroups.target_names[idx]:<20} {count:>4}个样本 ({count/len(data):.1%})")

# 特征工程
tfidf = TfidfVectorizer(max_features=5000)  # 限制特征数量加快训练
X_train_tfidf = tfidf.fit_transform(X_train)  # 训练集特征提取
X_test_tfidf = tfidf.transform(X_test)  # 测试集特征转换

# 朴素贝叶斯模型训练和评估
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)
print("\n朴素贝叶斯准确率:", accuracy_score(y_test, nb_pred))

# 随机森林模型训练和评估
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_pred = rf.predict(X_test_tfidf)
print("\n随机森林准确率:", accuracy_score(y_test, rf_pred))

# 安装SiliconFlow客户端
# 已经在requirements.txt中添加，运行pip install -r requirements.txt安装

# 设置API
client = OpenAI(
    api_key="YOUR_API_KEY_FROM_CLOUD_SILICONFLOW_CN",
    base_url="https://api.siliconflow.cn/v1"
)

# ICL提示模板设计


def build_prompt(train_texts, train_labels, test_text):
    """
    构建ICL(上下文学习)提示模板
    参数:
        train_texts: 训练文本列表
        train_labels: 对应标签列表
        test_text: 待预测文本
    返回:
        构造好的提示字符串
    """
    instruction = """请根据以下示例预测最后一个文本的类别标签(0/1/2)。
    只输出#数字#，不要包含其他内容。\n"""

    examples = ""
    for text, label in zip(train_texts, train_labels):
        examples += f"文本: {text[:200]}...\n类别: {label}\n\n"  # 截断长文本

    query = f"文本: {test_text[:200]}...\n类别:"

    return instruction + examples + query

# ICL实现函数


def run_llm_prediction(client, prompt, model="Qwen/Qwen2.5-7B-Instruct", temperature=0):
    """
    调用LLM进行预测
    参数:
        client: API客户端
        prompt: 构造好的提示
        model: 使用的模型名称
        temperature: 生成温度
    返回:
        LLM的响应内容
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用错误: {e}")
        return None

# 提取预测结果


def extract_prediction(response):
    """
    从LLM响应中提取预测结果
    参数:
        response: LLM的响应文本
    返回:
        提取的预测类别(0/1/2)或None(提取失败)
    """
    if response is None:
        return None

    # 尝试提取#数字#格式的结果
    match = re.search(r'#(\d+)#', response)
    if match:
        return int(match.group(1))
    else:
        # 尝试直接提取数字
        match = re.search(r'\d+', response)
        if match:
            return int(match.group(0))
        return None


# 为了避免API调用时触发速率限制，手动设置样本数量
n = 5  # 根据需要修改为 5，10, 20, 50 等

correct = 0  # 正确预测计数
for i in tqdm(range(len(X_test))):
    # 随机选择n个训练样本
    train_indices = np.random.choice(len(X_train), n, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"样本数量: {n}, ICL准确率: {accuracy:.4f}")

# 为了避免API调用时触发速率限制，手动设置样本数量
n = 10  # 根据需要修改为 5，10, 20, 50 等

correct = 0  # 正确预测计数
for i in tqdm(range(len(X_test))):
    # 随机选择n个训练样本
    train_indices = np.random.choice(len(X_train), n, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"样本数量: {n}, ICL准确率: {accuracy:.4f}")

# 为了避免API调用时触发速率限制，手动设置样本数量
n = 20  # 根据需要修改为 5，10, 20, 50 等

correct = 0  # 正确预测计数
for i in tqdm(range(len(X_test))):
    # 随机选择n个训练样本
    train_indices = np.random.choice(len(X_train), n, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"样本数量: {n}, ICL准确率: {accuracy:.4f}")

# 为了避免API调用时触发速率限制，手动设置样本数量
n = 50  # 根据需要修改为 5，10, 20, 50 等

correct = 0  # 正确预测计数
for i in tqdm(range(len(X_test))):
    # 随机选择n个训练样本
    train_indices = np.random.choice(len(X_train), n, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"样本数量: {n}, ICL准确率: {accuracy:.4f}")

# 测试不同版本模型的性能
# 手动设置模型名称，无需循环
model = "Qwen/Qwen2-7B-Instruct"

correct = 0
for i in tqdm(range(len(X_test))):
    # 使用固定数量的示例(10个)
    train_indices = np.random.choice(len(X_train), 10, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt, model=model)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print(f"模型: {model}, ICL准确率: {accuracy:.4f}")

# 测试不同版本模型的性能
# 手动设置模型名称，无需循环
model = "Qwen/Qwen2.5-7B-Instruct"

correct = 0
for i in tqdm(range(len(X_test))):
    # 使用固定数量的示例(10个)
    train_indices = np.random.choice(len(X_train), 10, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt, model=model)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print(f"模型: {model}, ICL准确率: {accuracy:.4f}")

# 测试不同大小模型的性能

model = "Qwen/Qwen2-1.5B-Instruct"
# 手动设置模型名称，无需循环

correct = 0
for i in tqdm(range(len(X_test))):
    # 使用固定数量的示例(10个)
    train_indices = np.random.choice(len(X_train), 10, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt, model=model)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print(f"模型: {model}, ICL准确率: {accuracy:.4f}")

# 测试不同大小模型的性能

model = "Qwen/Qwen2-7B-Instruct"
# 手动设置模型名称，无需循环

correct = 0
for i in tqdm(range(len(X_test))):
    # 使用固定数量的示例(10个)
    train_indices = np.random.choice(len(X_train), 10, replace=False)
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    response = run_llm_prediction(client, prompt, model=model)
    pred = extract_prediction(response)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print(f"模型: {model}, ICL准确率: {accuracy:.4f}")

# 定义不同的提示模板


def build_prompt_simple(train_texts, train_labels, test_text):
    """简单提示模板"""
    instruction = "预测类别: "
    examples = ""
    for text, label in zip(train_texts, train_labels):
        examples += f"'{text[:100]}': {label}\n"
    query = f"'{test_text[:100]}': "
    return instruction + examples + query


def build_prompt_detailed(train_texts, train_labels, test_text):
    # 详细提示
    instruction = """分析以下示例并分类新文本。
    示例中有不同类别的新闻文本(0=科学太空, 1=棒球运动, 2=中东政治)。
    只输出分类数字(0/1/2)，格式为: #数字#\n\n"""
    examples = ""
    for text, label in zip(train_texts, train_labels):
        examples += f"示例文本: {text[:150]}...\n分类: {label}\n\n"
    query = f"请分类:\n{test_text[:150]}...\n分类: "
    return instruction + examples + query


# 测试简单提示模板
name = "简单提示"
correct = 0  # 正确预测计数
for i in tqdm(range(len(X_test))):  # 遍历测试集
    # 随机选择10个训练样本作为示例
    train_indices = np.random.choice(len(X_train), 10, replace=False)
    # 构建提示
    prompt = build_prompt_simple(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    # 调用LLM进行预测
    response = run_llm_prediction(client, prompt)
    # 提取预测结果
    pred = extract_prediction(response)
    if pred == y_test[i]:  # 判断预测是否正确
        correct += 1

# 计算并直接打印准确率
accuracy = correct / len(X_test)
print(f"提示类型: {name}, ICL准确率: {accuracy:.4f}")

# 测试原始提示模板
name = "原始提示"
correct = 0  # 正确预测计数
for i in tqdm(range(len(X_test))):  # 遍历测试集
    # 随机选择10个训练样本作为示例
    train_indices = np.random.choice(len(X_train), 10, replace=False)
    # 构建提示
    prompt = build_prompt(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    # 调用LLM进行预测
    response = run_llm_prediction(client, prompt)
    # 提取预测结果
    pred = extract_prediction(response)
    if pred == y_test[i]:  # 判断预测是否正确
        correct += 1

# 计算并直接打印准确率
accuracy = correct / len(X_test)
print(f"提示类型: {name}, ICL准确率: {accuracy:.4f}")

# 测试详细提示模板
name = "详细提示"
correct = 0  # 正确预测计数
for i in tqdm(range(len(X_test))):  # 遍历测试集
    # 随机选择10个训练样本作为示例
    train_indices = np.random.choice(len(X_train), 10, replace=False)
    # 构建提示
    prompt = build_prompt_detailed(
        [X_train[j] for j in train_indices],
        [y_train[j] for j in train_indices],
        X_test[i]
    )
    # 调用LLM进行预测
    response = run_llm_prediction(client, prompt)
    # 提取预测结果
    pred = extract_prediction(response)
    if pred == y_test[i]:  # 判断预测是否正确
        correct += 1

# 计算并直接打印准确率
accuracy = correct / len(X_test)
print(f"提示类型: {name}, ICL准确率: {accuracy:.4f}")
