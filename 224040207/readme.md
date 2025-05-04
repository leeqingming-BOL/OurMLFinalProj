# Wine Classification: Logistic Regression vs DeepSeek ICL

This project compares traditional supervised learning (Logistic Regression) with in-context learning (ICL) using DeepSeek's large language model, on the Wine dataset from `sklearn.datasets`.

## Dataset

- **Wine dataset**: A classic multiclass classification dataset with 13 features describing the chemical properties of wines.
- 3 target classes: `class_0`, `class_1`, `class_2`

## Methods

### 1. Logistic Regression
- Standard supervised learning model
- Input features scaled using `StandardScaler`
- Trained on 80% of the data

### 2. DeepSeek In-Context Learning (ICL)
- Prompts are dynamically constructed using a few-shot approach
- Uses DeepSeek API to generate predictions
- Varies number of examples: 5, 10, 15
- Only the predicted class number (0, 1, 2) is returned

## Comparison Metrics

- **Accuracy**: Prediction accuracy on the test set
- **Time**: Time taken for inference
- **Valid Prediction Ratio**: Proportion of valid class predictions (DeepSeek only)

## How to Run

1. Install requirements:

   ```bash
   pip install numpy pandas scikit-learn openai
   ```

2. Replace your DeepSeek API key in the script:

   ```python
   API_KEY = "your_api_key_here"
   ```

3. Run the script:

   ```bash
   python your_script_name.py
   ```

## Sample Output

```
运行逻辑回归...
运行DeepSeek ICL...
DeepSeek ICL with 5 examples...
DeepSeek ICL with 10 examples...
DeepSeek ICL with 15 examples...

=== 结果比较 ===
Logistic Regression:
  准确率: 1.0000

DeepSeek-ICL-5ex:
  准确率: 0.9444
  预测时间: 162.12s
  有效预测比例: 100.00%

DeepSeek-ICL-10ex:
  准确率: 0.9444
  预测时间: 156.69s
  有效预测比例: 100.00%

DeepSeek-ICL-15ex:
  准确率: 0.9722
  预测时间: 167.54s
  有效预测比例: 100.00%
```

## Notes

* The DeepSeek API is called one sample at a time due to prompt-based inference.
* Model performance may vary depending on example selection and API behavior.




