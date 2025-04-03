# Credit Card Fraud Detection using Expectation-Maximization (EM) Algorithm

## Overview
This project applies the Expectation-Maximization (EM) algorithm using Gaussian Mixture Models (GMM) to detect fraudulent credit card transactions. The dataset is sourced from Kaggle and contains anonymized transaction features.

## Dataset
- **File:** `creditcard.csv`
- **Rows:** 284,807 transactions
- **Columns:** 31 features including `Time`, `Amount`, and `Class` (fraud or non-fraud)
- **Fraud Cases:** 492 fraudulent transactions (~0.172%)

## Data Preprocessing
1. **Check for missing values:** No missing values were found.
2. **Standardization:** The `Amount` column was standardized using `StandardScaler`.
3. **Feature Selection:** The `Time` column was dropped as it was not useful for EM.
4. **Dataset Splitting:** 80% for training, 20% for testing.

## Expectation-Maximization Algorithm
### **Implementation Steps:**
1. **Initialize Parameters:** Assume data follows a mixture of K Gaussian distributions (K=2, fraud vs. non-fraud).
2. **E-Step:** Compute probabilities of each transaction belonging to each cluster.
3. **M-Step:** Update Gaussian parameters (means, variances, and weights) to maximize likelihood.
4. **Repeat Until Convergence:** Iterate the E-step and M-step.

## Model Training
- Used `GaussianMixture` from `sklearn.mixture`.
- Tried different covariance types: `full`, `diag`, `tied`, `spherical`.
- Evaluated models using ROC-AUC scores.

## Model Evaluation
- Best performing model:
  - **Covariance Type:** `spherical`
  - **Number of Clusters:** `3`
  - **AUC Score:** `0.9559`

### **Comparison Across Covariance Types:**
| Covariance Type | Best AUC | Best Clusters |
|----------------|---------|--------------|
| Full          | 0.7129  | 2            |
| Diagonal (diag) | 0.9402  | 3            |
| Tied          | 0.6700  | 2            |
| **Spherical** | **0.9559** | **3**          |

## Key Findings
- The best model used **spherical covariance** with **3 clusters**, achieving the highest fraud detection performance.
- **Too many clusters (4+) reduce accuracy**, indicating overfitting.
- **Spherical and diagonal covariance performed best**, suggesting feature independence assumptions were valid.

## Conclusion
The Expectation-Maximization algorithm using Gaussian Mixture Models can effectively identify fraudulent transactions. Further improvements could involve feature engineering, dimensionality reduction (e.g., PCA), or hybrid supervised-unsupervised approaches.

## Requirements
- Python 3.x
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Scipy

## Usage
```python
# Load dataset
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

# Preprocess data
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df.drop(columns=["Time"], inplace=True)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='spherical', random_state=42)
gmm.fit(df.drop(columns=["Class"]))
```

## Acknowledgments
- Kaggle Credit Card Fraud Dataset.
- Scikit-learn's GaussianMixture Implementation.

## Future Work
- Improve feature selection.
- Compare with deep learning models.
- Deploy the model for real-time fraud detection.

