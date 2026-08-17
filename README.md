

##  1. Problem Statement
Determining whether a wild mushroom is safe to eat or highly toxic is a critical challenge requiring 100% predictive accuracy. The objective of this project is to build an automated, interpretable classification framework that evaluates different machine learning models to correctly classify mushrooms as either **Edible (e)** or **Poisonous (p)** based on their observable physical characteristics. 

By separating data preprocessing from application deployment and preserving model weights, this solution offers a fast, memory-efficient way to validate classification models on new test data without risking data leakage.

---

## 2. Dataset Description
This project utilizes the well-known **UCI Mushroom Dataset**. The dataset consists of records corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family.

* **Total Instances:** 8,124 mushrooms
* **Target Column:** `class` (Binary classification: `e` = Edible, `p` = Poisonous)
* **Features:** 22 categorical attributes describing physical features, including:
  * **Cap properties:** shape, surface, colour
  * **Stalk properties:** shape, root type, surface above/below ring, colour above/below ring
  * **Other identifiers:** bruises, odour, gill attachment, gill spacing, gill size, gill colour, veil type, veil colour, ring number, ring type, spore print colour, population, and habitat.

*Note: All features are entirely categorical strings, which are handled via an independent, pre-trained multi-column `OrdinalEncoder` mapping.*

---

## 3. GitHub Repository Link
Explore the source code, training pipelines, and deployment files here:  
**[GitHub Repository: mushroom-classification-dashboard](https://github.com/rajkumarvasan/bits-ml-assignment-2)** 

---

## 4. Models Used

1. **Random Forest Classifier**: An ensemble tree method providing high accuracy and feature importance rankings.
2. **Decision Tree Classifier**: A clear, interpretable flowchart style or Decision rule model used as a strong structure baseline.
3. **Logistic Regression**: A linear model optimized with high iterations to map categorical probability boundaries.
4. **K-Nearest Neighbors (KNN)**: A distance based classifier that categorizes samples based on the majority vote of their closest data points in the feature space.
5. **Naive Bayes (CategoricalNB)**: A probabilistic classifier based on Bayes' theorem, specifically optimized to handle independent categorical feature distributions.

---

### 5. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Decision Tree Classifier** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Logistic Regression** | 95.32% | 98.34% | 94.80% | 95.52% | 95.16% | 0.9064 |
| **K-Nearest Neighbors (KNN)** | 99.82% | 100.00% | 99.62% | 100.00% | 99.81% | 0.9963 |
| **Naive Bayes (CategoricalNB)** | 95.08% | 99.75% | 98.75% | 90.92% | 94.67% | 0.9038 |
| 

### 6. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves a solid 95.32% accuracy and 0.9064 MCC. While highly performant, it is restricted by its linear decision boundaries, missing roughly 4.68% of complex structural combinations within the mushroom feature mappings. |
| **Decision Tree** | Reaches a perfect 100.00% across all metrics (Accuracy, AUC, F1, and a 1.0000 MCC). This indicates the dataset possesses definitive rule-based conditional splits that a single tree maps flawlessly. |
| **kNN** | Delivers an outstanding 99.82% accuracy and a perfect 100.00% Recall score. This reveals that different mushroom classes occupy highly distinct, dense geometric clusters in the categorical feature space, enabling flawless matching with nearest neighbors. |
| **Naive Bayes** | Yields a high 99.75% AUC and strong 98.75% Precision, but suffers from the lowest Recall at 90.92% and the lowest MCC at 0.9038. The lower recall indicates that the strict feature independence assumption forces the model to make dangerous false negative errors. |
| **Random Forest (Ensemble)** | Achieves an absolute 100.00% on every single evaluation metric. By combining an ensemble of randomized trees through bagging, it completely eliminates variance and maps the categorical features without any risk of overfitting. |
| **Overall Winner for your dataset?** | **Random Forest Classifier** and **Decision Tree Classifier**. Both models natively handle categorical rule structures, achieve a perfect 100.00% accuracy, and gets an unmatched MCC score of 1.0000. |


