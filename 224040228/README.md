# ICL与传统机器学习对比实验

![综合对比图](/224040228/Results/en_comprehensive_comparison.png)

本项目对比研究了基于大语言模型(LLM)的上下文学习(ICL)与传统机器学习方法在文本分类任务上的表现。

## 项目概述

- **目标**：比较ICL与朴素贝叶斯、随机森林在20Newsgroups数据集上的分类性能
- **数据集**：选取了3个类别的20Newsgroups子集(sci.space, rec.sport.baseball, talk.politics.mideast)
- **模型**：
  - 传统方法：MultinomialNB, RandomForest
  - ICL方法：Qwen2和Qwen2.5系列不同规模的LLM(1.5B/7B参数)

## 主要发现

### 性能对比
| 方法 | 准确率 |
|------|--------|
| 朴素贝叶斯 | 92.3% |
| 随机森林 | 89.6% |
| ICL (最佳配置) | 76.2% |

![方法对比图](/224040228/Results/en_method_comparison.png)

### ICL关键影响因素
1. **示例数量**：5个示例(57.4%) → 50个示例(76.2%)
   ![示例数量影响图](/224040228/Results/en_sample_size_impact.png)

2. **模型规模**：1.5B模型(39.3%) < 7B模型(62-66%)
   ![模型对比图](/224040228/Results/en_model_comparison.png)

3. **提示工程**：不同提示模板准确率差异显著(30.4%-66.5%)
   ![提示模板对比图](/224040228/Results/en_prompt_comparison.png)

## 多维评估

![雷达对比图](/224040228/Results/en_radar_comparison.png)

## 文件结构

```
224040228/
├── icl_vs_traditional_ml.py      # 主实验代码
├── icl_vs_traditional_ml.ipynb   # Jupyter notebook版本
├── requirements.txt              # 依赖库
├── Results/                      # 实验结果图表
│   ├── en_sample_size_impact.png
│   ├── en_model_comparison.png
│   ├── en_prompt_comparison.png
│   ├── en_method_comparison.png
│   ├── en_comprehensive_comparison.png
│   ├── en_radar_comparison.png
│   └── resultsFigure.py          # 可视化代码
└── README.md                     # 本项目说明文件
```

## 使用说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行实验：
```bash
python icl_vs_traditional_ml.py
```

3. 生成可视化图表：
```bash
python Results/resultsFigure.py
```

4. 结果查看：
- 控制台输出各实验配置的准确率
- 自动生成`experiment_logs_<timestamp>.json`记录完整实验结果
- 在Results文件夹下生成可视化图表

## 关键代码功能

- `build_prompt()`: 构建ICL提示模板
- `run_llm_prediction()`: 调用LLM API进行预测
- `extract_prediction()`: 从LLM响应中提取预测结果
- `resultsFigure.py`: 生成所有实验结果可视化图表

## 后续研究方向

1. 扩展更多类型的数据集测试
2. 探索更复杂的提示工程技术
3. 分析ICL在小样本学习场景的优势
4. 比较计算资源消耗和推理速度

## 注意事项

1. 需要自行申请SiliconFlow API key并配置
2. 大规模实验可能受到API调用限制
3. 建议从少量示例开始测试
4. 可视化前请确保已安装matplotlib和seaborn