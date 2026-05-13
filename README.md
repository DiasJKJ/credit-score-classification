[alt text](image.png)
## 📂 Project Structure
- `DataProcessor`: Custom class for cleaning and tokenization.
- `ClipOutliersTransformer`: Custom transformer for handling financial outliers.
- `fit_model`: Wrapper for the training loop including early stopping.
- `credit_score_prediction_model-cb-v1`: The final saved model fileThis is a comprehensive overview of your project, incorporating the advanced CatBoost training logic and feature selection techniques from your notebook. You can use this for your GitHub `README.md`.

---

# Credit Score Classification with CatBoost 💳🤖

This project implements an intelligent credit scoring system using the **CatBoost** gradient boosting library. It automates the process of segregating customers into credit brackets (**Good**, **Standard**, **Poor**) based on financial behavior and historical bank details.

## 🌟 Key Features
- **Advanced Preprocessing:** Custom `ClipOutliersTransformer` for skewness correction and `IterativeImputer` for handling missing values.
- **Categorical & Text Handling:** Native support for categorical features and text-based tokenization of "Type of Loan" descriptions.
- **Feature Importance & Selection:** Systematic refinement of the model by removing low-impact features using the `LossFunctionChange` metric.
- **Production-Ready Training:** High-performance training using **GPU acceleration** and automated **class balancing**.



## 🏗️ Model Training Pipeline

The project follows a structured three-step training process:

### 1. Data Preparation (Pool Creation)
The `create_pool` function converts processed DataFrames into **CatBoost Pools**. This is an optimized data structure that handles categorical indices and text features internally, significantly speeding up the training process.

### 2. Model Configuration
The `catboost_model` function initializes a `CatBoostClassifier` with specific hyperparameters optimized for this dataset:
*   **Task Type:** `GPU` for rapid training.
*   **Text Processing:** `NaiveBayes+Word|BoW+Word` to extract features from loan types.
*   **Class Weights:** `Balanced` to handle uneven distributions of credit ratings.
*   **Regularization:** `depth=7` and `subsample=0.5` to prevent overfitting.

### 3. Feature Refinement
To ensure the model remains efficient and interpretable, a feature importance analysis was conducted. Features with an importance score below `0.00` were removed, leading to a "Refined Model" that maintains high accuracy with a leaner feature set.



## 📊 Performance Metrics

The refined model achieved a robust **80.62% Accuracy** on the test set.

| Metric    | Good   | Poor   | Standard | **Overall** |
|-----------|--------|--------|----------|-------------|
| **Precision** | 71%    | 78%    | 87%      | **81%**     |
| **Recall**    | 85%    | 86%    | 76%      | **81%**     |
| **F1-Score**  | 77%    | 82%    | 81%      | **81%**     |

### Confusion Matrix Insights
The model shows exceptional performance in identifying **Poor** and **Good** credit scores (high recall), which is crucial for risk management in financial institutions.