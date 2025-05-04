# Comparison Experiment between In-Context Learning (ICL) and Traditional Machine Learning

![Comprehensive Comparison Chart](/224040228/Results/en_comprehensive_comparison.png)

This project conducts a comparative study on the performance of in-context learning (ICL) based on large language models (LLMs) and traditional machine learning methods in text classification tasks.

## Project Overview

- **Objective**: Compare the classification performance of ICL, Naive Bayes, and Random Forest on the 20Newsgroups dataset.
- **Dataset**: A subset of the 20Newsgroups dataset with 3 categories (sci.space, rec.sport.baseball, talk.politics.mideast) is selected.
- **Models**:
  - Traditional methods: MultinomialNB, RandomForest
  - ICL methods: Different scales of LLMs in the Qwen2 and Qwen2.5 series (1.5B/7B parameters)

## Main Findings

### Performance Comparison
| Method | Accuracy |
| ------ | ------ |
| Naive Bayes | 92.3% |
| Random Forest | 89.6% |
| ICL (Best Configuration) | 76.2% |

![Method Comparison Chart](/224040228/Results/en_method_comparison.png)

### Key Influencing Factors of ICL
1. **Number of Examples**: From 5 examples (57.4%) to 50 examples (76.2%)
   ![Impact of Example Quantity Chart](/224040228/Results/en_sample_size_impact.png)

2. **Model Scale**: 1.5B model (39.3%) < 7B model (62 - 66%)
   ![Model Comparison Chart](/224040228/Results/en_model_comparison.png)

3. **Prompt Engineering**: Significant differences in accuracy among different prompt templates (30.4% - 66.5%)
   ![Prompt Template Comparison Chart](/224040228/Results/en_prompt_comparison.png)

## Multidimensional Evaluation

![Radar Comparison Chart](/224040228/Results/en_radar_comparison.png)

## File Structure

```
224040228/
├── icl_vs_traditional_ml.py      # Main experiment code
├── icl_vs_traditional_ml.ipynb   # Jupyter notebook version
├── requirements.txt              # Dependent libraries
├── Results/                      # Experiment result charts
│   ├── en_sample_size_impact.png
│   ├── en_model_comparison.png
│   ├── en_prompt_comparison.png
│   ├── en_method_comparison.png
│   ├── en_comprehensive_comparison.png
│   ├── en_radar_comparison.png
│   └── resultsFigure.py          # Visualization code
└── README.md                     # Project description file
```

## Usage Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the experiment:
```bash
python icl_vs_traditional_ml.py
```

3. Generate visualization charts:
```bash
python Results/resultsFigure.py
```

4. View the results:
- The console outputs the accuracy of each experimental configuration.
- Automatically generates `experiment_logs_<timestamp>.json` to record the complete experimental results.
- Visualization charts are generated in the Results folder.

## Key Code Functions

- `build_prompt()`: Build the ICL prompt template.
- `run_llm_prediction()`: Call the LLM API for prediction.
- `extract_prediction()`: Extract the prediction result from the LLM response.
- `resultsFigure.py`: Generate all experimental result visualization charts.

## Future Research Directions

1. Expand testing on more types of datasets.
2. Explore more complex prompt engineering techniques.
3. Analyze the advantages of ICL in small-sample learning scenarios.
4. Compare the computational resource consumption and inference speed.

## Notes

1. You need to apply for a SiliconFlow API key and configure it yourself.
2. Large-scale experiments may be restricted by API calls.
3. It is recommended to start testing with a small number of examples.
4. Make sure matplotlib and seaborn are installed before visualization.