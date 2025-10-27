
**0. Executive Summary**  
This project focused on building a predictive model for housing prices using a complex dataset with a mix of numerical and categorical features. Our primary goal was to develop a robust regression model that could generalize well to unseen data, despite facing substantial data quality and modeling challenges.
A major hurdle was the high proportion of missing values in critical columns such as cond_class, build_mat, and year_built. We addressed this through a hybrid imputation strategy combining KNN, mode, and mean imputation methods tailored to variable types and missingness levels. Feature engineering also played a central role; we created interaction terms between key numeric features (e.g., floor_max × floor_no, dist_post × dist_pharma) that exhibited high correlation with the target and added meaningful contextual information.
To ensure model stability and interpretability, we carefully filtered and evaluated categorical features using statistical techniques such as ANOVA and Cramér’s V, and removed over 18,000 numeric outliers using a z-score threshold. We standardized numeric features and one-hot encoded categorical ones via a preprocessing pipeline that preserved consistency across cross-validation folds.
We benchmarked several models, including Linear, Lasso, ElasticNet, and KNN regressors. While ElasticNet showed promise, Ridge Regression ultimately emerged as the most effective and stable approach. It offered strong predictive accuracy (R² = 0.95, RMSE ≈ 97,873) while effectively managing multicollinearity and preserving valuable information from weak but relevant features. Ridge’s simplicity and performance balance made it the optimal final model for this price prediction task.

**1. Challenges We Faced**

Challenge #1: Missing Data
One of the biggest issues we encountered was the high proportion of missing values in several important columns.  
For example:
- `cond_class`: 74.8% missing  
- `build_mat`: ~40% missing  
- `obj_type`: ~22% missing  
- `year_built`: ~16% missing  

Even structural and locational variables like `floor_no`, `infrastructure_quality`, and `has_lift` had noticeable gaps.  
This limited how much we could rely on these features and required us to develop a flexible and robust imputation strategy.

 
Challenge #2: High-Dimensional and Mixed-Type Features
The dataset included a large number of both numerical and categorical features, as well as many new interaction variables we created.  
Managing all of these in a clean and scalable way was difficult, especially when trying to prepare them for machine learning models without introducing sparsity or bias.

Challenge #3: Multicollinearity
After expanding the feature set with interactions (e.g., `floor_max × floor_no`, `dist_post × dist_pharma`),  
we noticed strong correlations between certain variables. This multicollinearity could have distorted model coefficients and hurt interpretability.

 
Challenge #4: Outliers in Numeric Features
We identified over 18,000 potential outliers in the numeric data using a z-score threshold of `|z| > 3`.  
These extreme values risked skewing the results and inflating errors, so we filtered them out before training our model.

Challenge #5: Weak or Redundant Features
Some categorical variables—like `obj_type`—didn’t provide much extra information and were also correlated with other fields.  
We used statistical methods like ANOVA and Cramér’s V to evaluate which ones were worth keeping and avoid overfitting due to high-cardinality encoding.

 	

Challenge #6: Weak Correlation with Target
Finally, we found that many raw features had only weak individual correlation with the target variable `price_z`.  
This meant we needed to rely more on creative feature engineering and interaction terms to build a stronger predictive signal.


**2.  How We Tackled It**

Handling Missing Values

We applied a hybrid imputation strategy based on missingness level and data type:

- Dropped Extremely Sparse Columns:  
  - Dropped `cond_class` due to >74% missing data.

- KNN Imputation for Categorical Features:  
  - For `build_mat` (~40%) and `obj_type` (~22%)  
  - Encoded with `OrdinalEncoder`, applied `KNNImputer (k=5)`, then decoded back.

- KNN Imputation for Structural Variables:  
  - For `floor_no`, `year_built`, and `infrastructure_quality`  
  - Used KNN with context features: `n_rooms`, `dim_m2`, `dist_centre`, etc.

- Mode Imputation:  
  - For binary/low-cardinality columns: `floor_max`, `has_lift`.

- Mean Imputation for Continuous Distances:  
  - For `dist_sch`, `dist_clinic`, `dist_uni`, `dist_post`, etc.  
  - These had <3% missing values and followed continuous distributions.

This approach helped preserve data while reducing risk of bias from naive imputation.

Feature Engineering & Variable Selection

We selected the top 20 numeric features most correlated with `price_z`, and from them, engineered multiple interaction features to capture complex relationships:

- `dim_m2 × n_rooms`  
- `floor_max × floor_no`  
- `dist_sch × dist_post`  
- Combinations of location-based distances, like `dist_post × dist_kind`

These new features captured contextual dependencies that helped the model better understand value influencers.

Selected interaction pairs:
dim_m2 x n_rooms (corr = 0.76)
dim_m2 x estimated_maintenance_cost (corr = 0.69)
src_month_year x src_month_month (corr = -0.89)
dist_centre x green_space_ratio (corr = -0.83)
floor_max x floor_no (corr = 0.68)
dist_sch x dist_post (corr = 0.74)
dist_sch x dist_pharma (corr = 0.78)
dist_sch x dist_kind (corr = 0.76)
dist_post x dist_pharma (corr = 0.78)
dist_post x dist_kind (corr = 0.71)
dist_pharma x dist_kind (corr = 0.82)
Categorical Variable Filtering

To avoid overfitting and reduce dimensionality:
- We used ANOVA F-tests to assess relationships between numeric and categorical variables.
- We used Cramér’s V to measure dependency between categorical features.

This helped us remove redundant variables and keep only the most informative categories.
    
  Variable  F-statistic        p-value
2    build_mat  5773.840845   0.000000e+00
0     obj_type  5125.797342   0.000000e+00
5     has_lift  5077.149503   0.000000e+00
7    has_store  3321.287730   0.000000e+00
8     loc_code  3219.617783   0.000000e+00
6      has_sec  2362.623929   0.000000e+00
3     has_park  2231.477909   0.000000e+00
1     own_type   725.174348  7.439744e-314
4  has_balcony   592.833589  1.223225e-130
Handling Outliers

We removed over 18,000 rows with any numeric variable where `|z-score| > 3`.  
This step stabilized our dataset and reduced noise before training.


Preprocessing Pipeline

We implemented a clean and consistent pipeline using `ColumnTransformer`:

- StandardScaler for numeric features  
- One-Hot Encoding for categorical features  

This ensured proper scaling and encoding across all folds of cross-validation, and helped us avoid data leakage.

**3.  Why We Chose Ridge Regression**

We evaluated multiple regression models before choosing Ridge as our final approach. Here's a summary of each model and why we ultimately selected Ridge:

 
Linear Regression
- Pros: Simple and interpretable.
- Cons: 
  - Very sensitive to multicollinearity and outliers.
  - Without regularization, it overfit the data due to the high dimensionality and interaction terms.
- Result: Lower test performance and instability in coefficients.

Lasso Regression
- Pros: Performs feature selection by shrinking some coefficients to zero.
- Cons: 
  - Eliminated too many weak-but-useful predictors.
  - Struggled with groups of correlated features (often picking one arbitrarily).
- Result: Less stable model with slightly worse performance than Ridge.

ElasticNet Regression
- Pros: Combines L1 and L2 regularization — balances between Lasso and Ridge.
- Cons: 
  - Performed well, but required careful tuning of two hyperparameters (`alpha` and `l1_ratio`).
  - Final performance was close to Ridge, but added complexity.
- Result: Good candidate, but Ridge was more stable and interpretable with similar accuracy.

K-Nearest Neighbors Regression (KNN)
- Pros: Non-parametric, no need for assumptions on data distribution.
- Cons: 
  - Computationally expensive with large datasets.
  - Sensitive to irrelevant features and feature scaling.
  - Struggled with high dimensionality and sparse data after encoding.
- Result: Lower R² and higher RMSE, especially on unseen data.


Why Ridge Regression

We chose Ridge Regression as our final model due to its:

- Robust handling of multicollinearity via L2 regularization.
- Retention of all features, including those weakly correlated with the target, which helped with generalization.
- Ease of tuning: only one hyperparameter (`alpha`), which we optimized using cross-validation.
- Strong performance on test data:
  - R²: 0.95  
  - RMSE: ≈ 97,873  
  - MAPE: ≈ 9.3%

