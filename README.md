

##  1. Problem Statement
Determining whether a wild mushroom is safe to eat or highly toxic is a critical challenge requiring 100% predictive accuracy. The objective of this project is to build an automated, interpretable classification framework that evaluates different machine learning models to correctly classify mushrooms as either **Edible (`e`)** or **Poisonous (`p`)** based on their observable physical characteristics. 

By separating data preprocessing from application deployment and preserving model weights, this platform offers a fast, memory-efficient way to validate classification models on new test data without risking data leakage or runtime crashes.

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
To ensure a robust comparative evaluation, five distinct machine learning classification algorithms are preserved as serialized `.joblib` files within the `model/` folder:

1. **Random Forest Classifier**: An ensemble tree method providing high accuracy and feature importance rankings.
2. **Decision Tree Classifier**: A clear, interpretable flowchart-style model used as a strong structure baseline.
3. **Logistic Regression**: A robust linear model optimized with high iterations to map categorical probability boundaries.
4. **K-Nearest Neighbors (KNN)**: A distance-based classifier that categorizes samples based on the majority vote of their closest data points in the feature space.
5. **Naive Bayes (CategoricalNB)**: A probabilistic classifier based on Bayes' theorem, specifically optimized to handle independent categorical feature distributions.

---

### 5. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Decision Tree Classifier** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Logistic Regression** | 96.55% | 99.41% | 96.59% | 96.51% | 96.55% | 0.9310 |
| **K-Nearest Neighbors (KNN)** | 99.85% | 99.98% | 99.85% | 99.85% | 99.85% | 0.9970 |
| **Naive Bayes (CategoricalNB)** | 95.81% | 98.92% | 95.95% | 95.73% | 95.80% | 0.9164 |
| 

### 6. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs very strongly (~96.5%) but falls slightly short of a perfect score due to its strict linearity assumption, which cannot fully capture complex multi-column interaction boundaries natively without manual feature engineering. |
| **Decision Tree** | Achieves a flawless 100% accuracy and AUC score. This dataset contains highly deterministic rule-based structures (e.g., specific combinations of 'odor' and 'gill-color' provide perfect splits), allowing a single tree to map the classes perfectly. |
| **kNN** | Yields near-perfect performance (~99.8%). Because poisonous and edible mushrooms exhibit tight, highly distinct visual and spatial cluster groups within the categorical attribute matrix, test instances match their nearest neighbors cleanly. |
| **Naive Bayes** | Obtains the lowest relative scores (~95.8%). This happens because the model relies on a strict assumption of conditional independence among all features, whereas mushroom characteristics (like stalk features and ring types) share strong natural correlations. |
| **Random Forest (Ensemble)** | Achieves a perfect 100% across all performance metrics. By combining multiple independent decision trees through bagging, it eliminates any individual tree variance and natively handles the categorical categorical mappings flawlessly. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** and **Decision Tree**. Both models handle categorical decision rules natively, score a perfect 100% across all metrics, and do not suffer from the independence or linearity limitations of the other models. |

