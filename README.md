

##  1. Problem Statement
It is very important to know whether a wild mushroom is safe to eat or poisonous, because a wrong prediction could be dangerous. The goal of this project is to build a machine learning system that can accurately classify mushrooms as **Edible (`e`)** or **Poisonous (`p`)** using their visible physical features.

To make the system efficient and reliable, data preprocessing is kept separate from model deployment, and the trained model weights are saved. This allows new test data to be checked quickly without data leakage or running into errors during execution.

---

## 2. Dataset Description
This project utilizes the well-known **UCI Mushroom Dataset**. The dataset consists of records corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family.

* **Total Instances:** 8,124 mushrooms
* **Target Column:** `class` (Binary classification: `e` = Edible, `p` = Poisonous)
* **Features:** 22 categorical attributes describing physical features, including:
  * **Cap properties:** shape, surface, colour
  * **Stalk properties:** shape, root type, surface above/below ring, colour above/below ring
  * **Other identifiers:** bruises, odour, gill attachment, gill spacing, gill size, gill colour, veil type, veil colour, ring number, ring type, spore print colour, population, and habitat.

*Note: All features are categorical strings, which are handled through an independent, pre-trained multi-column `OrdinalEncoder` mapping.*

---

## 3. GitHub Repository Link
Explore the source code, training pipelines, and deployment files here:  
**[GitHub Repository: mushroom-classification-dashboard](https://github.com/rajkumarvasan/bits-ml-assignment-2)** 

---

## 4. Models Used
To compare the models fairly, five machine learning classification algorithms are saved as `.joblib` files within the `model/` folder:

1. **Random Forest Classifier**: An ensemble model that usually gives high accuracy and shows which features matter most.
2. **Decision Tree Classifier**: A simple, easy-to-understand model that works like a flowchart.
3. **Logistic Regression**: A reliable linear model that is often used for classification problems.
4. **K-Nearest Neighbors (KNN)**: A model that classifies a sample based on the labels of its nearest neighbors.
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
| **Logistic Regression** | It performs well with 95.32% accuracy, but since it works best with straight-line patterns, it misses some of the more complex patterns in the data. |
| **Decision Tree** | It gets 100% on all metrics, which suggests the mushroom data has clear rule-based patterns that a single tree can learn easily. |
| **kNN** | 	It is almost perfect with 99.82% accuracy and identifies the target class correctly every time, showing that similar mushrooms are very close to each other in the dataset. |
| **Naive Bayes** | It has a high 99.75% AUC, but its Recall is only 90.92%. Because it assumes all features are independent, it sometimes makes unsafe mistakes and classifies poisonous mushrooms as edible. |
| **Random Forest (Ensemble)** | It achieves 100% across all scores. By combining many trees, it avoids small mistakes and handles the dataset very well.|
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** and **Decision Tree**. Both models got a perfect 100% score across the board because they are naturally great at handling categorical data rules. |

