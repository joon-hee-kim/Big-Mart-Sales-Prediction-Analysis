# Big-Mart-Sales-Prediction-Analysis

This is the result of an outlet-level sales prediction project using the Big Mart Sales Prediction dataset and machine learning-based data science methods.

## Motivation

Our team initiated this project to understand the end-to-end process of a data science project by predicting product sales at retail outlets. Using the Big Mart Sales Prediction dataset, we aimed to analyze how product-level and outlet-level features affect sales performance.

This project allowed us to apply a practical data science workflow, including data exploration, missing-value handling, categorical encoding, outlier detection, feature scaling, feature selection, model training, and model evaluation. Through this process, we gained hands-on experience in applying machine learning models to a structured real-world dataset.

## End-to-End Process (with Output)

## 1. Business Objective </br>

- Predict outlet-level product sales using multiple product and outlet features. </br>
- Analyze which product-level and outlet-level features are related to `Item_Outlet_Sales`. </br>

## 2. Data Exploration </br>

- Loaded the Big Mart Sales Prediction dataset using `pd.read_csv()`. </br>
- Printed the first five rows of the dataset using `df.head(5)`. </br>
- Checked the dataset structure using `df.shape`, `df.index`, and `df.columns`. </br>
<img width="662" height="294" alt="image" src="https://github.com/user-attachments/assets/c8b5a225-1949-467a-a35f-ff7ef0683874" /> </br>

- Generated statistical summaries using `df.describe()`. </br>
<img width="426" height="228" alt="image" src="https://github.com/user-attachments/assets/456da2b5-a234-46a1-97b5-c2b6ad3e16be" /> </br>

- Checked feature names and data types using `df.dtypes`. </br>
<img width="426" height="272" alt="image" src="https://github.com/user-attachments/assets/8f011470-4568-4d32-995a-019be5f61516" /> </br>

- The dataset includes product-level and outlet-level features such as: </br>
  - `Item_Identifier` </br>
  - `Item_Weight` </br>
  - `Item_Fat_Content` </br>
  - `Item_Visibility` </br>
  - `Item_Type` </br>
  - `Item_MRP` </br>
  - `Outlet_Identifier` </br>
  - `Outlet_Establishment_Year` </br>
  - `Outlet_Size` </br>
  - `Outlet_Location_Type` </br>
  - `Outlet_Type` </br>
  - `Item_Outlet_Sales` </br>

- `Item_Identifier` and `Outlet_Identifier` were excluded because they are unique identification numbers and do not directly contribute to the prediction task. </br>

- Non-numerical columns were encoded for visualization and analysis: </br>
  - Ordinal Encoding: `Item_Fat_Content`, `Outlet_Location_Type`, `Outlet_Size` </br>
  - Label Encoding: `Item_Type` </br>
  - One-Hot Encoding: `Outlet_Type` </br>

- Histograms were generated to visualize feature distributions. </br>

<img width="810" height="430" alt="image" src="https://github.com/user-attachments/assets/bc348339-b7b4-442b-b873-97e26f5d12d6" /> </br>

## 3. Data Preprocessing </br>

### 3.1 Missing-Value Handling </br>

- Filled missing values in `Item_Weight` with the mean value. </br>
- Filled missing values in `Outlet_Size` with `Medium`. </br>

### 3.2 Dropping Unnecessary Columns </br>

- Removed columns that were not used for the final modeling process. </br>
- The dropped columns were: </br>
  - `Item_Identifier` </br>
  - `Item_Fat_Content` </br>
  - `Outlet_Identifier` </br>
  - `Outlet_Size` </br>

### 3.3 Item Type Category Reduction </br>

- Reclassified `Item_Type` values into broader categories to simplify the feature space. </br>
- The reduced categories were: </br>
  - `Desserts` </br>
  - `Stored foods` </br>
  - `Carbohydrates` </br>
  - `Fats` </br>
  - `Drinks` </br>
  - `Other foods` </br>
  - `Etc` </br>

### 3.4 One-Hot Encoding </br>

- Applied One-Hot Encoding to the `Outlet_Type` column. </br>
- Converted `Outlet_Type` into new columns: </br>
  - `Outlet_Type_Grocery Store` </br>
  - `Outlet_Type_Supermarket Type1` </br>
  - `Outlet_Type_Supermarket Type2` </br>
  - `Outlet_Type_Supermarket Type3` </br>

## 4. Outlier Detection and Removal </br>

- Drew boxplots for numerical columns to check outliers. </br>
- The checked numerical columns were: </br>
  - `Item_Weight` </br>
  - `Outlet_Establishment_Year` </br>
  - `Item_MRP` </br>
  - `Item_Outlet_Sales` </br>

- Removed outliers from the `Item_Outlet_Sales` column using the Interquartile Range (IQR) method. </br>
- After removing outliers, a boxplot was generated again to check the result. </br>

#### Before Outlier Removal </br>
<img width="664" height="463" alt="image" src="https://github.com/user-attachments/assets/b308d318-7b12-4db7-b1b1-608093988817" /> </br>

#### After Outlier Removal </br>
<img width="244" height="244" alt="image" src="https://github.com/user-attachments/assets/f1cf6071-3dd1-4263-8502-bc5e6acac31c" /> </br>

## 5. Encoding and Scaling </br>

### 5.1 Encoding </br>

- Applied Label Encoding to `Item_Type`. </br>
- Applied Ordinal Encoding to `Outlet_Location_Type`. </br>

#### Before Encoding </br>
<img width="594" height="272" alt="image" src="https://github.com/user-attachments/assets/c7ff3bfc-ba5e-42cd-af68-b87fd67e4259" /> </br>

#### After Encoding </br>
<img width="372" height="271" alt="image" src="https://github.com/user-attachments/assets/e4e717a5-3797-4402-8073-8775de1e15ee" /> </br>

### 5.2 Scaling </br>

- Applied RobustScaler to `Item_Weight`. </br>
- Applied MinMaxScaler to `Outlet_Establishment_Year`. </br>
- Applied StandardScaler to `Item_MRP`. </br>

#### Before Scaling </br>
<img width="612" height="230" alt="image" src="https://github.com/user-attachments/assets/bd4ff134-2051-4c55-8237-055daf22f567" /> </br>

#### After Scaling </br>
<img width="454" height="240" alt="image" src="https://github.com/user-attachments/assets/5acc63bc-e7c3-465f-829d-e1935b333077" /> </br>

## 6. Feature Selection </br>

- Separated the dataset into independent features and the dependent target variable. </br>
- The target variable was `Item_Outlet_Sales`. </br>
- Visualized the correlation between features using a heatmap. </br>

<img width="614" height="480" alt="image" src="https://github.com/user-attachments/assets/16a00fff-1891-49e5-a30e-b308e8b8fdf4" /> </br>

- Used `RandomForestRegressor` to calculate feature importance. </br>

- The top important features identified from Random Forest feature importance were: </br>
  - `Item_MRP` </br>
  - `Outlet_Type_Grocery Store` </br>
  - `Outlet_Type_Supermarket Type3` </br>
  - `Item_Visibility` </br>

<img width="586" height="316" alt="image" src="https://github.com/user-attachments/assets/97dd8bee-3219-4e67-b73b-068bec74017a" /> </br>

## 7. Modeling </br>

- Trained a Linear Regression model to predict `Item_Outlet_Sales`. </br>
- Split the dataset into training and testing sets using `train_test_split`. </br>

## 8. Model Evaluation </br>

- Evaluated the Linear Regression model using K-Fold Cross-Validation. </br>
- Used Mean Squared Error (MSE) and R-Squared (R2) as evaluation metrics. </br>

- Visualized the distribution of actual and predicted sales values using KDE plots. </br>

<img width="602" height="302" alt="image" src="https://github.com/user-attachments/assets/e0a3c41a-8f88-4f64-ab56-7e41f335085c" /> </br>
<img width="613" height="197" alt="image" src="https://github.com/user-attachments/assets/fd20cf02-ceb0-4358-b418-8cddaa377f4f" /> </br>


## 9. Learning Experience </br>

Through this project, we learned how to apply machine learning models to a real-world structured dataset. We also experienced the full end-to-end process of a data science project, including data exploration, preprocessing, feature selection, model training, evaluation, result interpretation, and visualization. </br>

### Difficulties </br>

- Different categorical features required different encoding techniques. </br>
- Model performance had to be evaluated using appropriate metrics. </br>
- The model needed to be evaluated in a way that checks whether it generalizes to unseen data. </br>

### Solutions </br>

- Used OrdinalEncoder for `Outlet_Location_Type`. </br>
- Used LabelEncoder for `Item_Type`. </br>
- Used OneHotEncoder for `Outlet_Type`. </br>
- Used K-Fold Cross-Validation with MSE and R2 to evaluate model performance. </br>

## 10. Open Source SW </br>

- Created a GitHub repository to manage and share the project. </br>
- Used Python libraries and machine learning tools for data processing, visualization, modeling, and evaluation. </br>
- Tested model evaluation by changing K-Fold settings, test size, and encoding type. </br>

### Main Libraries </br>

- `pandas` </br>
- `numpy` </br>
- `seaborn` </br>
- `matplotlib` </br>
- `scikit-learn` </br>

### Main Functions and Methods </br>

- `pd.read_csv()` — Load the dataset. </br>
- `df.head()` — Print the first rows of the dataset. </br>
- `df.describe()` — Generate statistical summaries. </br>
- `df.dtypes` — Check data types. </br>
- `df.fillna()` — Fill missing values. </br>
- `df.drop()` — Drop unnecessary columns. </br>
- `OneHotEncoder().fit_transform()` — Apply one-hot encoding. </br>
- `LabelEncoder().fit_transform()` — Apply label encoding. </br>
- `OrdinalEncoder().fit_transform()` — Apply ordinal encoding. </br>
- `RobustScaler()` — Scale numerical features robustly. </br>
- `MinMaxScaler()` — Normalize numerical features. </br>
- `StandardScaler()` — Standardize numerical features. </br>
- `sns.boxplot()` — Detect outliers visually. </br>
- `sns.heatmap()` — Visualize feature correlations. </br>
- `RandomForestRegressor()` — Calculate feature importance. </br>
- `LinearRegression()` — Train the regression model. </br>
- `train_test_split()` — Split data into training and testing sets. </br>
- `KFold()` — Perform K-Fold Cross-Validation. </br>
- `cross_validate()` — Evaluate the model with multiple metrics. </br>
- `sns.kdeplot()` — Visualize actual and predicted value distributions. </br>

## 👥 Team Member

201934219 Kim Joonhee </br>
201935025 Kim Sion </br>
202035393 Ju Yongwan </br>
202135759 Kim Hyunjeong </br>

## ✔️ Source

* ChatGPT </br>
* [Big Mart Sales Prediction Dataset](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction) </br>
* [scikit-learn Documentation](https://scikit-learn.org/stable/) </br>
* [Seaborn Documentation](https://seaborn.pydata.org/) </br>
* [Dimensionality Reduction Techniques for Categorical and Continuous Data](https://medium.com/codex/dimensionality-reduction-techniques-for-categorical-continuous-data-75d2bca53100) </br>
* [Regression Model Evaluation Metrics](https://velog.io/@ljs7463/%ED%9A%8C%EA%B7%80%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9Cevaluation-metrics) </br>
* [One-Hot Encoding: pd.get_dummies vs OneHotEncoder](https://coduking.tistory.com/entry/%EC%9B%90-%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9-pdgetdummies-vs-OneHotEncoder) </br>
* [Additional Reference Site 1](https://static.vecteezy.com/system/resources/previews/034/908/550/non_2x/software-architecture-line-icon-illustration-vector.jpg) </br>
* [Additional Reference Site 2](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5dIOXIQKtRejMkbE9DEYGd4trRUCXcyRrulSI_uO9I2rvi4LDaVA724d-BeNipJq9GOU&usqp=CAU) </br>
* [Additional Reference Site 3](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_6ZO29yw6XC_Cs14GToX3OlbMfDPBdzG38V19M_n0g6mR56FllJiplCPaoyEPEvuiFJs&usqp=CAU) </br>
* [Additional Reference Site 4](https://www.shareicon.net/download/2015/09/25/107141_network.svg) </br>
* [Additional Reference Site 5](https://m.blog.naver.com/youji4ever/221712578078) </br>
* [Additional Reference Site 6](https://dlearner.tistory.com/20) </br>
* [Additional Reference Site 7](https://gmnam.tistory.com/302) </br>
* [Additional Reference Site 8](https://wikibook.co.kr/ml-definitive-guide/) </br>
