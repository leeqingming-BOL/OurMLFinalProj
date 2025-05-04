# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import requests
import json
import time



# 1. 数据准备
def prepare_data():
    # 加载wine数据集
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    # 划分训练测试集 (80%训练，20%测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'target_names': target_names
    }


# 2. 逻辑回归实现
def run_logistic_regression(data, C=1.0):
    # 使用标准化后的数据
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(data['X_train_scaled'], data['y_train'])

    # 预测
    y_pred = model.predict(data['X_test_scaled'])
    accuracy = accuracy_score(data['y_test'], y_pred)

    return {
        'model': model,
        'accuracy': accuracy,
        'y_pred': y_pred
    }


# 3. DeepSeek ICL实现
from openai import OpenAI

class DeepSeekICL:
    def __init__(self, api_key, model_name="deepseek-chat", base_url="https://api.deepseek.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def create_prompt(self, data, example_count=5):
        prompt = "请根据以下葡萄酒的化学特征预测其类别。类别有：0='class_0', 1='class_1', 2='class_2'。\n\n"
        for i in range(min(example_count, len(data['X_train']))):
            features = ", ".join(
                [f"{name}={value:.2f}" for name, value in zip(data['feature_names'], data['X_train'][i])])
            prompt += f"示例{i + 1}: 特征=[{features}], 类别={data['y_train'][i]}\n"
        prompt += "\n请只回答数字类别（0、1或2），不要包含其他文本。\n"
        return prompt

    def predict(self, data, example_count=5, temperature=0):
        y_pred = []
        prompt_template = self.create_prompt(data, example_count)

        for x in data['X_test']:
            features = ", ".join([f"{name}={value:.2f}" for name, value in zip(data['feature_names'], x)])
            full_prompt = prompt_template + f"\n请预测: 特征=[{features}], 类别="

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=temperature,
                    max_tokens=10
                )
                prediction = response.choices[0].message.content.strip()
                try:
                    pred_class = int(prediction[0])
                    y_pred.append(pred_class)
                except:
                    y_pred.append(-1)

                time.sleep(0.1)

            except Exception as e:
                print(f"API请求失败: {e}")
                y_pred.append(-1)

        valid_preds = [pred for pred in y_pred if pred != -1]
        valid_true = [true for pred, true in zip(y_pred, data['y_test']) if pred != -1]

        accuracy = accuracy_score(valid_true, valid_preds) if valid_preds else 0.0
        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'valid_ratio': len(valid_preds) / len(y_pred)
        }


# 4. 对比分析
def compare_models(data, api_key):
    results = {}

    # 运行逻辑回归
    print("运行逻辑回归...")
    lr_results = run_logistic_regression(data, C=1.0)
    results['Logistic Regression'] = {
        'accuracy': lr_results['accuracy'],
        'time': None,  # 可以添加时间测量
        'y_pred': lr_results['y_pred']
    }

    # 运行DeepSeek ICL
    print("运行DeepSeek ICL...")
    ds_icl = DeepSeekICL(api_key)

    # 尝试不同示例数量
    for example_count in [5, 10, 15]:
        print(f"DeepSeek ICL with {example_count} examples...")
        start_time = time.time()
        icl_results = ds_icl.predict(data, example_count=example_count)
        elapsed_time = time.time() - start_time

        results[f'DeepSeek-ICL-{example_count}ex'] = {
            'accuracy': icl_results['accuracy'],
            'time': elapsed_time,
            'valid_ratio': icl_results['valid_ratio'],
            'y_pred': icl_results['y_pred']
        }

    return results


# 主函数
def main():
    # 准备数据
    data = prepare_data()

    API_KEY = "sk-1c49a2772b8e4f32abf50591a5c5a14c"

    # 比较模型
    results = compare_models(data, API_KEY)

    # 打印结果
    print("\n=== 结果比较 ===")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        if 'time' in metrics and metrics['time'] is not None:
            print(f"  预测时间: {metrics['time']:.2f}s")
        if 'valid_ratio' in metrics:
            print(f"  有效预测比例: {metrics['valid_ratio']:.2%}")
        print()


if __name__ == "__main__":
    main()