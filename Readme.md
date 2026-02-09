# 🏠 California Housing Price Prediction

An end-to-end machine learning project to predict median house values in California districts using the California Housing dataset. This project follows the workflow from **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** by Aurélien Géron.

---

## 📋 Project Overview

This project demonstrates a complete machine learning pipeline including:
- Data acquisition and exploration
- Data preprocessing and feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Final model deployment

The goal is to predict the **median house value** for California districts based on features like location, housing characteristics, and demographics.

---

## 📁 Project Structure

```
Machine Learning Housing Corporation/
├── data/
│   ├── train.csv              # Training dataset (80%)
│   ├── test.csv               # Test dataset (20%)
│   └── housing_cleaned.csv    # Preprocessed dataset (optional)
├── datasets/
│   └── housing/
│       └── housing.csv        # Original raw dataset
├── images/                    # Generated visualizations
├── models/                    # Saved models
├── 1-Download_Data.py         # Download and explore raw data
├── 2-Plots.py                 # Plotting utilities and settings
├── 4-Split_train-test.py      # Stratified train-test split
├── 6-Data_cleaning (1).py     # Manual data cleaning exploration
├── 6-Data_cleaning (2).py     # Pipeline-based preprocessing
├── 7-training.py              # Model training and evaluation
├── my_california_housing_model.pkl  # Final saved model
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy matplotlib joblib
```

### Running the Pipeline

1. **Download the data:**
   ```bash
   python 1-Download_Data.py
   ```

2. **Split into train/test sets (stratified by income):**
   ```bash
   python 4-Split_train-test.py
   ```

3. **Train and evaluate models:**
   ```bash
   python 7-training.py
   ```

---

## 📊 Dataset

The California Housing dataset contains **20,640 districts** with the following features:

| Feature | Description |
|---------|-------------|
| `longitude` | District longitude |
| `latitude` | District latitude |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | District population |
| `households` | Number of households |
| `median_income` | Median income (in tens of thousands USD) |
| `ocean_proximity` | Location relative to ocean (categorical) |
| **`median_house_value`** | **Target variable** (median house value in USD) |

---

## 🔧 Data Preprocessing Pipeline

The preprocessing pipeline applies the following transformations:

### 1. Missing Value Imputation
- **Numerical features:** Imputed with **median** values
- **Categorical features:** Imputed with **most frequent** category

### 2. Feature Engineering
Created new ratio features that better correlate with house prices:
- `bedrooms_ratio` = total_bedrooms / total_rooms
- `rooms_per_house` = total_rooms / households
- `people_per_house` = population / households

### 3. Geographic Clustering
- Used **KMeans clustering** on latitude/longitude
- Computed **RBF kernel similarity** to cluster centers
- This captures location-based patterns better than raw coordinates

### 4. Log Transformation
Applied logarithm to features with long tails:
- `total_bedrooms`, `total_rooms`, `population`, `households`, `median_income`

### 5. Categorical Encoding
- **One-Hot Encoding** for `ocean_proximity`
- Categories: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`

### 6. Feature Scaling
- **StandardScaler** applied to all numerical features

---

## 🤖 Models Evaluated

| Model | Training RMSE | Cross-Validation RMSE (10-fold) |
|-------|---------------|--------------------------------|
| Linear Regression | 68,648 | ~ 68,600 |
| Decision Tree | 0 (overfitting!) | ~ 69,000 |
| **Random Forest** | ~ 18,000 | **46,938** |

The Decision Tree achieves 0 RMSE on training data because it memorizes the training set (severe overfitting). Cross-validation reveals its true performance.

---

## 🎯 Hyperparameter Tuning

Used **RandomizedSearchCV** with 3-fold cross-validation to tune:

| Hyperparameter | Search Range | Best Value |
|----------------|--------------|------------|
| `n_clusters` (geographic) | 3 - 50 | **45** |
| `max_features` (Random Forest) | 2 - 20 | **9** |

---

## 📈 Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| **Test RMSE** | **$41,556** |
| 95% Confidence Interval | Computed using t-distribution |

### Top Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `median_income` (log) | 0.1888 |
| 2 | `ocean_proximity_INLAND` | 0.0755 |
| 3 | `bedrooms_ratio` | 0.0643 |
| 4 | `rooms_per_house_ratio` | 0.0522 |
| 5 | `people_per_house_ratio` | 0.0466 |
| 6 | Geographic Cluster 3 | 0.0424 |
| 7 | Geographic Cluster 17 | 0.0233 |
| 8 | Geographic Cluster 18 | 0.0226 |

**Key Insights:**
- **Median income** is by far the most predictive feature (~19% importance)
- Being **INLAND** (away from the coast) significantly affects prices
- The **engineered ratio features** are more important than raw counts
- **Geographic clustering** captures location-based price patterns effectively

---

## 💾 Model Persistence

The final trained model is saved using `joblib`:

```python
import joblib

# Save
joblib.dump(final_model, "my_california_housing_model.pkl")

# Load and use
model = joblib.load("my_california_housing_model.pkl")
predictions = model.predict(new_data)
```

The saved model includes the full preprocessing pipeline, so you can directly predict on raw data without manual preprocessing.

---

## 🔬 Custom Transformer: ClusterSimilarity

A custom Scikit-Learn transformer was created to compute geographic cluster similarities:

```python
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, 
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
        
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
```

This transformer:
1. Clusters geographic coordinates using KMeans
2. Transforms each district to its RBF similarity to each cluster center
3. Districts near high-value areas (e.g., Bay Area) get high similarity to those clusters

---

## 📚 What I Learned

1. **End-to-end ML workflow** from data acquisition to model deployment
2. **Scikit-Learn pipelines** for reproducible preprocessing
3. **Feature engineering** (ratios, log transforms, clustering)
4. **Cross-validation** to get reliable performance estimates
5. **Hyperparameter tuning** with RandomizedSearchCV
6. **Custom transformers** that integrate with Scikit-Learn
7. **Model persistence** with joblib

---

## 📖 References

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
- [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
- [Scikit-Learn Documentation](https://scikit-learn.org/)

---
