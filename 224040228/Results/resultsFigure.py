# -*- coding: utf-8 -*-
# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties

# =============================================================================
# 中文可视化部分
# =============================================================================


def plot_chinese_figures():
    """生成所有中文图表"""

    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    # 设置图表风格
    plt.style.use('ggplot')

    # 1. 样本数量对ICL准确率的影响
    sample_sizes = [5, 10, 20, 50]
    accuracies = [0.5744, 0.6410, 0.7145, 0.7624]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, accuracies, marker='o',
             linewidth=2, markersize=10, color='#1f77b4')
    plt.xlabel('上下文学习示例数量', fontsize=14)
    plt.ylabel('ICL准确率', fontsize=14)
    plt.title('示例样本数量对ICL准确率的影响', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sample_sizes)
    plt.yticks(np.arange(0.5, 0.85, 0.05))

    # 添加数据标签
    for x, y in zip(sample_sizes, accuracies):
        plt.text(x, y+0.01, f'{y:.4f}', ha='center', va='bottom', fontsize=12)

    # 添加趋势线拟合
    z = np.polyfit(sample_sizes, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(sample_sizes, p(sample_sizes), "r--", alpha=0.7)

    plt.tight_layout()
    plt.savefig('cn_sample_size_impact.png', dpi=300)
    plt.close()

    # 2. 不同模型的ICL准确率对比
    models = ['Qwen2-1.5B-Instruct',
              'Qwen2-7B-Instruct', 'Qwen2.5-7B-Instruct']
    model_accuracies = [0.3932, 0.6325, 0.6376]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, model_accuracies, color=[
                   '#ff9999', '#66b3ff', '#99ff99'], width=0.5)
    plt.xlabel('模型', fontsize=14)
    plt.ylabel('ICL准确率', fontsize=14)
    plt.title('不同模型ICL准确率对比', fontsize=16, fontweight='bold')
    plt.ylim(0, 0.7)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('cn_model_comparison.png', dpi=300)
    plt.close()

    # 3. 不同提示模板的ICL准确率对比
    prompts = ['简单提示', '原始提示', '详细提示']
    prompt_accuracies = [0.3043, 0.6650, 0.3829]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(prompts, prompt_accuracies, color=[
                   '#c2c2f0', '#ffcc99', '#c2f0c2'], width=0.5)
    plt.xlabel('提示模板类型', fontsize=14)
    plt.ylabel('ICL准确率', fontsize=14)
    plt.title('不同提示模板ICL准确率对比', fontsize=16, fontweight='bold')
    plt.ylim(0, 0.7)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('cn_prompt_comparison.png', dpi=300)
    plt.close()

    # 4. 最优ICL设置与传统机器学习方法的对比
    methods = ['朴素贝叶斯', '随机森林', 'ICL (50示例)']
    method_accuracies = [0.9231, 0.8957, 0.7624]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(methods, method_accuracies, color=[
                   '#ff9999', '#66b3ff', '#99ff99'], width=0.5)
    plt.xlabel('方法', fontsize=14)
    plt.ylabel('准确率', fontsize=14)
    plt.title('传统机器学习与最优ICL设置性能对比', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('cn_method_comparison.png', dpi=300)
    plt.close()

    # 5. 综合对比图：所有实验结果
    plt.figure(figsize=(15, 10))

    # 创建数据
    categories = ['传统方法', '样本数量', '模型规模', '提示模板']
    group_labels = [
        ['朴素贝叶斯', '随机森林'],
        ['5示例', '10示例', '20示例', '50示例'],
        ['1.5B模型', '7B模型', '7B+模型'],
        ['简单提示', '原始提示', '详细提示']
    ]
    all_values = [
        [0.9231, 0.8957],
        [0.5744, 0.6410, 0.7145, 0.7624],
        [0.3932, 0.6325, 0.6376],
        [0.3043, 0.6650, 0.3829]
    ]
    colors = [['#ff9999', '#ff7777'],
              ['#66b3ff', '#4499ff', '#3388ff', '#2277ff'],
              ['#99ff99', '#77ff77', '#55ff55'],
              ['#ffcc99', '#ffbb77', '#ffaa55']]

    # 设置位置
    x_positions = []
    x_ticks = []
    x_ticks_pos = []
    width = 0.7
    gap = 0.8

    current_pos = 0
    for i, group in enumerate(all_values):
        group_pos = []
        for j in range(len(group)):
            group_pos.append(current_pos)
            current_pos += width
        x_positions.append(group_pos)

        # 添加组标签位置
        x_ticks_pos.append(sum(group_pos) / len(group_pos))
        current_pos += gap

    # 绘制柱状图
    for i, (group_pos, group_vals, group_colors) in enumerate(zip(x_positions, all_values, colors)):
        for j, (pos, val, color) in enumerate(zip(group_pos, group_vals, group_colors)):
            bar = plt.bar(pos, val, width=width, color=color,
                          edgecolor='black', linewidth=1)
            plt.text(pos, val + 0.01, f'{val:.4f}',
                     ha='center', va='bottom', fontsize=10)

    # 设置标签和标题
    plt.ylabel('准确率', fontsize=14)
    plt.title('DDA5001项目实验结果综合对比', fontsize=18, fontweight='bold')
    plt.xticks(x_ticks_pos, categories, fontsize=12, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 添加图例
    legend_handles = []
    legend_labels = []
    for i, (group_label, group_colors) in enumerate(zip(group_labels, colors)):
        for j, (label, color) in enumerate(zip(group_label, group_colors)):
            patch = plt.Rectangle((0, 0), 1, 1, color=color,
                                  edgecolor='black', linewidth=1)
            legend_handles.append(patch)
            legend_labels.append(label)

    plt.legend(legend_handles, legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=10)

    plt.tight_layout()
    plt.savefig('cn_comprehensive_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 创建雷达图比较最优ICL与传统方法
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # 评估指标
    metrics = ['准确率', '速度', '灵活性', '解释性', '无监督能力']
    N = len(metrics)

    # 评分 (1-5分)
    # 朴素贝叶斯: 高准确率, 高速度, 低灵活性, 中等解释性, 低无监督能力
    nb_values = [4.6, 5.0, 2.0, 3.0, 1.5]
    # 随机森林: 较高准确率, 中等速度, 中等灵活性, 较高解释性, 低无监督能力
    rf_values = [4.5, 3.0, 3.0, 4.0, 1.5]
    # ICL: 中等准确率, 低速度, 高灵活性, 低解释性, 高无监督能力
    icl_values = [3.8, 1.5, 4.5, 2.0, 4.5]

    # 角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形

    # 扩展数据使图形闭合
    nb_values += nb_values[:1]
    rf_values += rf_values[:1]
    icl_values += icl_values[:1]

    # 绘制雷达图
    ax.plot(angles, nb_values, linewidth=2,
            linestyle='solid', label='朴素贝叶斯', color='#ff9999')
    ax.plot(angles, rf_values, linewidth=2,
            linestyle='solid', label='随机森林', color='#66b3ff')
    ax.plot(angles, icl_values, linewidth=2, linestyle='solid',
            label='ICL (50示例)', color='#99ff99')

    # 填充区域
    ax.fill(angles, nb_values, alpha=0.25, color='#ff9999')
    ax.fill(angles, rf_values, alpha=0.25, color='#66b3ff')
    ax.fill(angles, icl_values, alpha=0.25, color='#99ff99')

    # 设置雷达图坐标标签
    plt.xticks(angles[:-1], metrics, fontsize=12)

    # 设置y轴刻度范围
    ax.set_rlim(0, 5)
    plt.yticks([1, 2, 3, 4, 5], ['很差', '较差', '一般', '较好', '很好'], fontsize=10)
    plt.ylim(0, 5)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('传统机器学习与ICL方法多维度评估', fontsize=16, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig('cn_radar_comparison.png', dpi=300)
    plt.close()

    print("中文图表生成完成!")

# =============================================================================
# 英文可视化部分
# =============================================================================


def plot_english_figures():
    """Generate all English figures"""

    # Reset matplotlib params
    plt.rcParams.update(plt.rcParamsDefault)

    # Set figure style
    plt.style.use('ggplot')

    # 1. Impact of Sample Size on ICL Accuracy
    sample_sizes = [5, 10, 20, 50]
    accuracies = [0.5744, 0.6410, 0.7145, 0.7624]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, accuracies, marker='o',
             linewidth=2, markersize=10, color='#1f77b4')
    plt.xlabel('Number of ICL Examples', fontsize=14)
    plt.ylabel('ICL Accuracy', fontsize=14)
    plt.title('Impact of Example Count on ICL Accuracy',
              fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sample_sizes)
    plt.yticks(np.arange(0.5, 0.85, 0.05))

    # Add data labels
    for x, y in zip(sample_sizes, accuracies):
        plt.text(x, y+0.01, f'{y:.4f}', ha='center', va='bottom', fontsize=12)

    # Add trend line fitting
    z = np.polyfit(sample_sizes, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(sample_sizes, p(sample_sizes), "r--", alpha=0.7)

    plt.tight_layout()
    plt.savefig('en_sample_size_impact.png', dpi=300)
    plt.close()

    # 2. Different Model Comparison for ICL Accuracy
    models = ['Qwen2-1.5B-Instruct',
              'Qwen2-7B-Instruct', 'Qwen2.5-7B-Instruct']
    model_accuracies = [0.3932, 0.6325, 0.6376]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, model_accuracies, color=[
                   '#ff9999', '#66b3ff', '#99ff99'], width=0.5)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('ICL Accuracy', fontsize=14)
    plt.title('ICL Accuracy Comparison Across Models',
              fontsize=16, fontweight='bold')
    plt.ylim(0, 0.7)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('en_model_comparison.png', dpi=300)
    plt.close()

    # 3. Different Prompt Template Comparison
    prompts = ['Simple Prompt', 'Default Prompt', 'Detailed Prompt']
    prompt_accuracies = [0.3043, 0.6650, 0.3829]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(prompts, prompt_accuracies, color=[
                   '#c2c2f0', '#ffcc99', '#c2f0c2'], width=0.5)
    plt.xlabel('Prompt Template Type', fontsize=14)
    plt.ylabel('ICL Accuracy', fontsize=14)
    plt.title('ICL Accuracy Comparison Across Prompt Templates',
              fontsize=16, fontweight='bold')
    plt.ylim(0, 0.7)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('en_prompt_comparison.png', dpi=300)
    plt.close()

    # 4. Best ICL Setting vs Traditional ML Methods
    methods = ['Naive Bayes', 'Random Forest', 'ICL (50 examples)']
    method_accuracies = [0.9231, 0.8957, 0.7624]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(methods, method_accuracies, color=[
                   '#ff9999', '#66b3ff', '#99ff99'], width=0.5)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Performance Comparison: Traditional ML vs Best ICL Setting',
              fontsize=16, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('en_method_comparison.png', dpi=300)
    plt.close()

    # 5. Comprehensive Comparison: All Experimental Results
    plt.figure(figsize=(15, 10))

    # Create data
    categories = ['Traditional Methods', 'Sample Size',
                  'Model Scale', 'Prompt Templates']
    group_labels = [
        ['Naive Bayes', 'Random Forest'],
        ['5 examples', '10 examples', '20 examples', '50 examples'],
        ['1.5B model', '7B model', '7B+ model'],
        ['Simple Prompt', 'Default Prompt', 'Detailed Prompt']
    ]
    all_values = [
        [0.9231, 0.8957],
        [0.5744, 0.6410, 0.7145, 0.7624],
        [0.3932, 0.6325, 0.6376],
        [0.3043, 0.6650, 0.3829]
    ]
    colors = [['#ff9999', '#ff7777'],
              ['#66b3ff', '#4499ff', '#3388ff', '#2277ff'],
              ['#99ff99', '#77ff77', '#55ff55'],
              ['#ffcc99', '#ffbb77', '#ffaa55']]

    # Set positions
    x_positions = []
    x_ticks = []
    x_ticks_pos = []
    width = 0.7
    gap = 0.8

    current_pos = 0
    for i, group in enumerate(all_values):
        group_pos = []
        for j in range(len(group)):
            group_pos.append(current_pos)
            current_pos += width
        x_positions.append(group_pos)

        # Add group label positions
        x_ticks_pos.append(sum(group_pos) / len(group_pos))
        current_pos += gap

    # Draw bar chart
    for i, (group_pos, group_vals, group_colors) in enumerate(zip(x_positions, all_values, colors)):
        for j, (pos, val, color) in enumerate(zip(group_pos, group_vals, group_colors)):
            bar = plt.bar(pos, val, width=width, color=color,
                          edgecolor='black', linewidth=1)
            plt.text(pos, val + 0.01, f'{val:.4f}',
                     ha='center', va='bottom', fontsize=10)

    # Set labels and title
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('DDA5001 Project: Comprehensive Experimental Results',
              fontsize=18, fontweight='bold')
    plt.xticks(x_ticks_pos, categories, fontsize=12, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Add legend
    legend_handles = []
    legend_labels = []
    for i, (group_label, group_colors) in enumerate(zip(group_labels, colors)):
        for j, (label, color) in enumerate(zip(group_label, group_colors)):
            patch = plt.Rectangle((0, 0), 1, 1, color=color,
                                  edgecolor='black', linewidth=1)
            legend_handles.append(patch)
            legend_labels.append(label)

    plt.legend(legend_handles, legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=10)

    plt.tight_layout()
    plt.savefig('en_comprehensive_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Create radar chart comparing optimal ICL with traditional methods
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Evaluation metrics
    metrics = ['Accuracy', 'Speed', 'Flexibility',
               'Interpretability', 'Zero-shot Capability']
    N = len(metrics)

    # Ratings (1-5 scale)
    # Naive Bayes: high accuracy, high speed, low flexibility, medium interpretability, low zero-shot
    nb_values = [4.6, 5.0, 2.0, 3.0, 1.5]
    # Random Forest: high accuracy, medium speed, medium flexibility, high interpretability, low zero-shot
    rf_values = [4.5, 3.0, 3.0, 4.0, 1.5]
    # ICL: medium accuracy, low speed, high flexibility, low interpretability, high zero-shot
    icl_values = [3.8, 1.5, 4.5, 2.0, 4.5]

    # Angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the plot

    # Extend data to close the plot
    nb_values += nb_values[:1]
    rf_values += rf_values[:1]
    icl_values += icl_values[:1]

    # Draw radar chart
    ax.plot(angles, nb_values, linewidth=2, linestyle='solid',
            label='Naive Bayes', color='#ff9999')
    ax.plot(angles, rf_values, linewidth=2, linestyle='solid',
            label='Random Forest', color='#66b3ff')
    ax.plot(angles, icl_values, linewidth=2, linestyle='solid',
            label='ICL (50 examples)', color='#99ff99')

    # Fill areas
    ax.fill(angles, nb_values, alpha=0.25, color='#ff9999')
    ax.fill(angles, rf_values, alpha=0.25, color='#66b3ff')
    ax.fill(angles, icl_values, alpha=0.25, color='#99ff99')

    # Set radar chart coordinate labels
    plt.xticks(angles[:-1], metrics, fontsize=12)

    # Set y-axis scale range
    ax.set_rlim(0, 5)
    plt.yticks([1, 2, 3, 4, 5], ['Very Poor', 'Poor',
               'Average', 'Good', 'Excellent'], fontsize=10)
    plt.ylim(0, 5)

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Multi-dimensional Evaluation: Traditional ML vs ICL Methods',
              fontsize=16, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig('en_radar_comparison.png', dpi=300)
    plt.close()

    print("English figures generated successfully!")


# =============================================================================
# 主函数
# =============================================================================
if __name__ == "__main__":
    # 生成中文图表
    plot_chinese_figures()

    # 生成英文图表
    plot_english_figures()

    print("所有图表生成完成! All figures have been generated!")
