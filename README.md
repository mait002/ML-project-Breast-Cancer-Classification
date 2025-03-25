# Breast Cancer Classification using Decision Trees (Scikit-learn)

This project focuses on building a binary classification model using the **Breast Cancer Wisconsin dataset** to predict whether a tumor is benign or malignant based on diagnostic features. The model is developed using Scikit-learn’s **Decision Tree Classifier** with a focus on interpretability, model tuning, and visualization.

---

## 📊 Dataset

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features:** 30 numeric features (e.g., mean radius, texture, concavity)
- **Target Classes:**
  - `0` = Malignant
  - `1` = Benign
- **Instances:** 569

---

## 🔍 Project Highlights

### ✔️ Data Preprocessing
- Loaded data using `load_breast_cancer(return_X_y=True)`
- Stratified train-test split to preserve class distribution (60/40)

### 🌲 Model Building
- Used `DecisionTreeClassifier` with:
  - **Entropy** as the split criterion
  - `min_samples_split=6` to control tree complexity
- Visualized decision tree using `plot_tree` with feature and class names

### 📈 Evaluation & Tuning
- Trained and tested the model over a range of **tree depths (1–15)**
- Tracked training and testing accuracy to observe overfitting behavior
- Used `GridSearchCV` for **hyperparameter tuning** of `max_depth` and `min_samples_split` with 5-fold cross-validation
- Best model: tree depth of `3`, balanced accuracy on unseen data

---

## 📌 Key Learnings

- Stratified sampling ensures fair class representation across folds
- Tree depth has a significant impact on model generalization
- GridSearchCV provides a robust way to optimize hyperparameters while avoiding overfitting
- Visualization helps interpret model decisions and understand feature importance

---

## 📂 Folder Structure

```plaintext
├── breast_cancer_decision_tree.ipynb   # Main notebook
├── README.md
