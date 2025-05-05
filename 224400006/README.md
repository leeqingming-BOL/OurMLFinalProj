# ICL and ConvNet for optical digit recognition
The following project presents a classification problem which tests the multimodal capability of LLMs for the purpose of digit recognition. An ICL method is applied, and the performance is compared to that of a statistical learning method (ConvNet)

## Dataset
The dataset used for this task is 'Optical Recognition of Handwritten Digits' from the UC Irvine Machine Learning Depository ([link](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)). It contains 5620 instances of handwritten digits 0-9, represented as 8x8 matrices with elements valued 0..16 based on the intensity of the "square". 
- No. of features: 8x8 = 64 features
- Multiclass problem: 10 classes (digits 0-9)

## Learning models
Note: there are two separate notebooks, one for each model. 
- ConvNet: the statistical learning model of choice for this task is a ConvNet with 2 convolutional layers, a fully connected layer and softmax activation function. 
- LLM: the large-language model of choice is Qwen2.5-VL-72B-Instruct, a model in Qwen's 'Visual-Language' series that promises good performance on visual tasks and recognition, on par with "...GPT-4o and Claude 3.5 Sonnet, particularly excelling in document and diagram understanding" ([Qwen team, 2025](https://arxiv.org/abs/2502.13923))

## Objective
The two models are trained on the data and are then tested to predict digits of test samples with the goal of achieving the highest Accuracy score (= True predictions/All predictions).

## Instructions
1. Install required libraries:
```bash
pip install torch/sklearn/numpy/pandas/matplotlib/seaborn/openai
```
2. Fetch your personal API key from [SiliconFlow](https://siliconflow.cn/zh-cn/) and replace the key in *llm_digit_recog.ipynb*:
```python
API_KEY = "your_key_here"
```
3. Run the notebooks

## Sample Outputs
- Training the ConvNet
```bash
Epoch 1/20, Train Loss: 2.1157, Train Acc: 36.32%, Test Loss: 1.4974, Test Acc: 70.55%
Epoch 2/20, Train Loss: 0.8261, Train Acc: 78.78%, Test Loss: 0.4220, Test Acc: 88.61%
Epoch 3/20, Train Loss: 0.3617, Train Acc: 88.77%, Test Loss: 0.2478, Test Acc: 93.15%
Epoch 4/20, Train Loss: 0.2635, Train Acc: 92.08%, Test Loss: 0.1983, Test Acc: 94.31%
Epoch 5/20, Train Loss: 0.2037, Train Acc: 94.02%, Test Loss: 0.1673, Test Acc: 95.73%
Epoch 6/20, Train Loss: 0.1603, Train Acc: 95.13%, Test Loss: 0.1506, Test Acc: 96.00%
Epoch 7/20, Train Loss: 0.1432, Train Acc: 95.71%, Test Loss: 0.1182, Test Acc: 96.26%
Epoch 8/20, Train Loss: 0.1248, Train Acc: 96.15%, Test Loss: 0.1175, Test Acc: 96.62%
```

- Inference with the LLM
```
Processing sample 40/100
Digit to be identified:
      ░░██      
      ▒▒██      
    ▒▒██▒▒░░    
    ██▓▓░░██░░  
  ░░██░░▓▓██▒▒  
    ▓▓██████░░  
        ██▒▒    
        ██▒▒    

Sample 40: Predicted 4, Actual 4
Processing sample 41/100
Digit to be identified:
    ▓▓▓▓▓▓░░    
    ▓▓██▓▓██░░  
    ▒▒██▓▓██    
    ░░████░░    
    ▓▓████      
    ██▒▒▓▓▒▒    
  ░░██░░▓▓██    
  ░░▓▓██▓▓░░    

Sample 41: Predicted 8, Actual 8
```

The final outputs of both notebooks contain the accuracy, as well as confusion matrices.