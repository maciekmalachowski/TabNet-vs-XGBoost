# TabNet-vs-XGBoost

<img src="banner.png" alt="TabNet vs XGBoost">

When it comes to tabular data, **XGBoost** has long been a dominant machine learning algorithm. However, in recent years, **TabNet**, a deep learning architecture specifically designed for tabular data, has emerged as a strong contender. In this blog post, we'll explore both algorithms by comparing their performance on various tasks and examine the surprising strengths of TabNet.

<h2>What is <a href="https://github.com/dreamquark-ai/tabnet" target="_blank" style="text-decoration:none;">TabNet</a> ?</h2>

**TabNet** was proposed by researchers at **Google Cloud** in 2019 to bring the power of deep learning to tabular data. Despite the rise of neural networks in fields like image processing, natural language understanding, and speech recognition, tabular data—which is still the foundation of many industries like healthcare, finance, retail, and marketing—has traditionally been dominated by tree-based models like **XGBoost**.

The motivation behind TabNet is to leverage deep learning's proven ability to generalize well on large datasets. Unlike tree-based models, which do not efficiently optimize errors using techniques like **Gradient Descent**, deep neural networks have the potential to adapt more effectively through continuous learning. TabNet specifically addresses this by incorporating a **sequential attention** mechanism, allowing the model to selectively focus on the most relevant features during training. This not only improves performance but also enhances interpretability, as it is easy to see which features are driving the predictions.

By combining these principles, **TabNet** delivers a high-performance deep learning architecture optimized for tabular data, providing both accuracy and interpretability.

<br>

## Installing TabNet

You can install **TabNet** simply using pip or conda as follows.

- with **pip**  
```python
pip install pytorch-tabnet
```

- with **conda**
```python
conda install -c conda-forge pytorch-tabnet
```

<br>

## Datasets

For this comparison, I generated synthetic datasets using scikit-learn's `make_classification` and `make_regression` functions. These functions allow us to create customizable datasets for binary classification, multiclass classification, and regression tasks.

Here’s a quick overview of the datasets:

- **Binary Classification Dataset**: Generated using make_classification with 10,000 samples, 100 features, and 8 informative features. This dataset has 2 classes.

```python
from sklearn.datasets import make_classification
X_binary, y_binary = make_classification(n_samples=10000, n_features=100, n_informative=8, n_classes=2, random_state=42)
```

- **Multiclass Classification Dataset**: Also generated using make_classification, but this time with 3 classes. The dataset contains 10,000 samples, 100 features, and 8 informative features, and we set n_clusters_per_class=1 to have more distinct class clusters.

```python
from sklearn.datasets import make_classification
X_multiclass, y_multiclass = make_classification(n_samples=10000, n_features=100, n_informative=8, n_classes=3, n_clusters_per_class=1, random_state=42)
```

- **Regression Dataset**: This dataset was generated using make_regression, with 10,000 samples and 100 features. We defined 8 informative features and added a bit of noise to simulate a more realistic setting.

```python
from sklearn.datasets import make_regression
X_regression, y_regression = make_regression(n_samples=10000, n_features=100, n_informative=8, noise=0.1, random_state=42)
```
These datasets are designed to test both algorithms across different tasks and complexities, ensuring that the results are both reliable and replicable.

<br>

## Performance Comparison
To compare TabNet and XGBoost, I conducted a series of experiments using the datasets above. First, I used the default parameters for both models to observe their initial performance.

**1. Binary Classification (Default)**

```python
TabNet Accuracy: 0.896
XGBoost Accuracy: 0.912
```

**2. Multiclass Classification (Default)**

```python
TabNet Accuracy: 0.928
XGBoost Accuracy: 0.944
```

**3. Regression (Default)**

```python
TabNet RMSE: 11.1389
XGBoost RMSE: 43.011
```

### Initial Observations

From the results above, we can see that XGBoost slightly outperforms TabNet in classification tasks (binary and multiclass) when using default parameters. However, the regression results show a significant difference in favor of TabNet, where it drastically outperformed XGBoost in terms of RMSE. This result is particularly interesting because XGBoost has traditionally been favored for regression tasks, yet here we see TabNet excelling.

<br>

## Fine-Tuning and Custom Parameter Adjustments

Next, I adjusted the algorithms with custom parameters to see how much improvement could be achieved with proper tuning. 

For **TabNet**, I used custom parameters such as:
- `eval_metric` = Accuracy (Classification) / RMSE (Regression)
- `max_epochs` = 100
- `patience` = 30
- `optimizer_params` = 0.09
```python
# Classification
    clf_tabnet = TabNetClassifier(optimizer_params=dict(lr=0.09), verbose=0)

    clf_tabnet.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["accuracy"],
        max_epochs=100,
        patience=30
    )

# Regression
    reg_tabnet = TabNetRegressor(optimizer_params=dict(lr=0.09), verbose=0)

    reg_tabnet.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["rmse"],
        max_epochs=100,
        patience=30
    )
```

For **XGBoost**, the following parameters were adjusted:
- `eval_metric` = Logloss (Classification) / RMSE (Regression)
- `n_estimators` = 1000
- `early_stopping_rounds` = 30
- `learning_rate` = 0.05
```python
# Classification
    clf_xgboost = xgb.XGBClassifier(n_estimators=1000, eval_metric="logloss", early_stopping_rounds=30, learning_rate=0.05)

    clf_xgboost.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

# Regression
    reg_xgboost = xgb.XGBRegressor(n_estimators=1000, eval_metric="rmse", early_stopping_rounds=30, learning_rate=0.05)
    
    reg_xgboost.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
```

Interestingly, while testing different parameter values for both algorithms, the **default versions of some parameters of XGBoost and TabNet performed best** in most cases. Fine-tuning provided marginal improvements in accuracy and RMSE, but the default configurations already showcased solid performance.

**1. Binary Classification (Tuned)**
```python
TabNet Accuracy: 0.9593
XGBoost Accuracy: 0.92
```

**2. Multiclass Classification (Tuned)**
```python
TabNet Accuracy: 0.9647
XGBoost Accuracy: 0.9387
```

**3. Regression (Tuned)**
```python
TabNet RMSE: 8.4831
XGBoost RMSE: 31.1758
```

### Key Insights from Fine-Tuning

**Binary Classification**: After fine-tuning, **TabNet significantly outperformed XGBoost**, achieving an accuracy of **0.9593** compared to XGBoost's **0.92**. This shows that with the right tuning, TabNet can deliver much stronger results even in traditionally tree-dominated areas like classification.

**Multiclass Classification**: Here, **TabNet outperformed XGBoost**, reaching an accuracy of **0.9647**, while XGBoost achieved **0.9387**. TabNet's ability to handle multiple classes effectively makes it a powerful tool for multiclass tasks, especially with proper parameter adjustment.

**Regression**: **TabNet continued to dominate** in the regression task. After fine-tuning, its RMSE dropped to **8.4831**, compared to XGBoost's RMSE of **31.1758**. This stark contrast highlights TabNet’s ability to capture complex relationships in continuous data, making it especially useful for regression tasks where interpretability and attention to important features matter.

<br>

## Conclusion: TabNet’s Surprising Strengths

In this comparison, both **TabNet** and **XGBoost** demonstrated excellent performance across tasks, especially after tuning. However, what stands out is **TabNet's superior performance in both classification and regression** tasks. In particular, TabNet dramatically outperformed XGBoost in the regression task, challenging the common assumption that gradient-boosting algorithms like XGBoost are the best option for regression on tabular data.

Moreover, the fact that **TabNet consistently surpassed XGBoost in both binary and multiclass classification tasks** after fine-tuning shows that it is not just competitive, but often better-suited for a wide range of tabular data problems.

What’s particularly exciting about TabNet is its integration of **deep learning principles like attention mechanisms**, combined with a focus on interpretability. This makes it a modern and powerful tool for handling tabular datasets. Its ability to deliver strong results without extensive feature engineering, especially in regression, makes TabNet a model worth considering for many real-world applications.
