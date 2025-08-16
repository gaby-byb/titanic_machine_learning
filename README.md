# 🚢 Titanic Survival Prediction – KNN Model

This project uses the Titanic dataset to predict passenger survival using a **K-Nearest Neighbors (KNN)** machine learning model.  
It includes full **data cleaning**, **feature engineering**, **scaling**, **hyperparameter tuning**, and **model evaluation** inside a Jupyter Notebook.

---

## Files in This Repo
- `titanic_knn.ipynb` → Jupyter Notebook with all code + explanations
- `titanic.csv` → Dataset (Titanic passenger info)
- `.gitignore` → Keeps Jupyter checkpoints and cache out of Git


## Steps in the Notebook
1. **Imports** → Pull in pandas, NumPy, scikit-learn, Matplotlib, Seaborn
2. **Load Data** → Read Titanic CSV, check for missing values
3. **Preprocessing** → Drop irrelevant columns, fill missing data, convert text to numbers
4. **Feature Engineering** → Add `FamilySize`, `IsAlone`, `FareBin`, `AgeBin`
5. **Scaling** → Normalize features to 0–1 range
6. **Hyperparameter Tuning** → GridSearchCV to find best KNN settings
7. **Evaluation** → Accuracy score + confusion matrix


## Model Details
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Tuning Parameters**: 
  - Number of neighbors (1–20)
  - Distance metric (Euclidean, Manhattan, Minkowski)
  - Weights (uniform, distance)
- **Validation**: 5-fold cross-validation


##Example Output
Accuracy: 82.50%
Confusion Matrix:
[[105 14]
[ 18 62]]

*(Numbers will vary slightly depending on train/test split)*


## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/gaby-byb/titanic-knn.git
   cd titanic-knn
2. Install requirements
   pip install pandas numpy scikit-learn matplotlib seaborn
