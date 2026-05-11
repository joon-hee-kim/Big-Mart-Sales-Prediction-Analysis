# Big-Mart-Sales-Prediction-Analysis

This is the result of an outlet-level sales prediction project using the Big Mart Sales Prediction dataset and machine learning-based data science methods.

## Motivation

Our team initiated this project to understand the end-to-end process of a data science project by predicting product sales at retail outlets. Using the Big Mart Sales Prediction dataset, we aimed to analyze how product-level and outlet-level features affect sales performance.

This project allowed us to apply a practical data science workflow, including data exploration, missing-value handling, categorical encoding, outlier detection, feature scaling, feature selection, model training, and model evaluation. Through this process, we gained hands-on experience in applying machine learning models to a structured real-world dataset.

## End-to-End Process (with Output)

## 1. Business Objective </br>

- Predict outlet-level product sales using multiple product and outlet features. </br>
- Understand which features are important for predicting `Item_Outlet_Sales`. </br>
- Apply an end-to-end data science workflow from data exploration to evaluation. </br>

<!-- Add output image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="750"> </br> -->

## 2. Data Exploration </br>

- Loaded the Big Mart Sales Prediction dataset using `pd.read_csv()`. </br>
- Printed the first five rows of the dataset using `df.head(5)`. </br>
- Checked the dataset structure using `df.shape`, `df.index`, and `df.columns`. </br>
- Generated statistical summaries using `df.describe()`. </br>
- Checked feature names and data types using `df.dtypes`. </br>

<!-- Add output image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="750"> </br> -->

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
  - Ordinal Encoding: `Item_Fat_Content` </br>
  - Label Encoding: `Item_Type` </br>
  - Ordinal Encoding: `Outlet_Location_Type`, `Outlet_Size` </br>
  - One-Hot Encoding: `Outlet_Type` </br>

- Histograms were generated to visualize feature distributions. </br>

<!-- Add histogram output image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="850"> </br> -->

## 3. Data Preprocessing </br>

### 3.1 Missing-Value Handling </br>

- Filled missing values in `Item_Weight` with the mean value. </br>
- Filled missing values in `Outlet_Size` with `Medium`. </br>

```python
df = df.fillna({'Item_Weight': df['Item_Weight'].mean()})
df = df.fillna({'Outlet_Size': 'Medium'})

```

### 3.2 Dropping Unnecessary Columns </br>

- Removed columns that were not used for the final modeling process. </br>
- The dropped columns were: </br>
  - `Item_Identifier` </br>
  - `Item_Fat_Content` </br>
  - `Outlet_Identifier` </br>
  - `Outlet_Size` </br>

```python
drop_groups = ['Item_Identifier', 'Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Size']
df = df.drop(columns=drop_groups)
```

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

```python
def get_category(type):
  cat = ''
  if type == "Fruits and Vegetables" and "Snack Foods":
    cat = 'Desserts'
  elif type == "Frozen Foods" or type == "Canned":
    cat = 'Stored foods'
  elif type == "Breads" or type == "Starchy Foods":
    cat = 'Carbohydrates'
  elif type == "Meat" or type == "Dairy":
    cat = 'Fats'
  elif type == "Soft Drinks" or type == "Hard Drinks":
    cat = 'Drinks'
  elif type == "Breakfast" or type == "Seafood":
    cat = 'Other foods'
  else:
    cat = 'Etc'
  return cat

df['Item_Type'] = df['Item_Type'].apply(lambda x: get_category(x))
```

### 3.4 One-Hot Encoding </br>

- Applied One-Hot Encoding to the `Outlet_Type` column. </br>
- Converted `Outlet_Type` into new columns: </br>
  - `Outlet_Type_Grocery Store` </br>
  - `Outlet_Type_Supermarket Type1` </br>
  - `Outlet_Type_Supermarket Type2` </br>
  - `Outlet_Type_Supermarket Type3` </br>

```python
label = df['Outlet_Type']

ohe = OneHotEncoder(sparse_output=False)
outlet_type_encoded = ohe.fit_transform(label.values.reshape(-1, 1))
outlet_type_encoded_df = pd.DataFrame(
    outlet_type_encoded,
    columns=ohe.get_feature_names_out(['Outlet_Type'])
)

df = pd.concat([df, outlet_type_encoded_df], axis=1)
df = df.drop(columns=['Outlet_Type'])
```

## 4. Outlier Detection and Removal </br>

- Drew boxplots for numerical columns to check outliers. </br>
- The checked numerical columns were: </br>
  - `Item_Weight` </br>
  - `Outlet_Establishment_Year` </br>
  - `Item_MRP` </br>
  - `Item_Outlet_Sales` </br>

```python
whis_value = 2.0

plt.figure(figsize=(10, 7))
for i, column in enumerate(
    ['Item_Weight', 'Outlet_Establishment_Year', 'Item_MRP', 'Item_Outlet_Sales'], 1
):
    plt.subplot(2, 3, i)
    sns.boxplot(df[column], whis=whis_value)
    plt.title(column)

plt.tight_layout()
plt.show()
```

- Removed outliers from the `Item_Outlet_Sales` column using the Interquartile Range (IQR) method. </br>
- After removing outliers, a boxplot was generated again to check the result. </br>

```python
column = 'Item_Outlet_Sales'

Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - whis_value * IQR
upper_bound = Q3 + whis_value * IQR

outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
df.drop(outliers.index, inplace=True)

print("Number of rows removed for Item_Outlet_Sales:", len(outliers))
```

<!-- Add boxplot output image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="750"> </br> -->

## 5. Encoding and Scaling </br>

### 5.1 Encoding </br>

- Applied Label Encoding to `Item_Type`. </br>
- Applied Ordinal Encoding to `Outlet_Location_Type`. </br>

```python
le = LabelEncoder()
df['Item_Type'] = le.fit_transform(df['Item_Type'])

oe = OrdinalEncoder()
df['Outlet_Location_Type'] = oe.fit_transform(df[['Outlet_Location_Type']])
```

### 5.2 Scaling </br>

- Applied RobustScaler to `Item_Weight`. </br>
- Applied MinMaxScaler to `Outlet_Establishment_Year`. </br>
- Applied StandardScaler to `Item_MRP`. </br>

```python
df['Item_Weight'] = RobustScaler().fit_transform(df[['Item_Weight']])
df['Outlet_Establishment_Year'] = MinMaxScaler().fit_transform(df[['Outlet_Establishment_Year']])
df['Item_MRP'] = StandardScaler().fit_transform(df[['Item_MRP']])
```

<!-- Add scaling output image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="850"> </br> -->

## 6. Feature Selection </br>

- Separated the dataset into independent features and the dependent target variable. </br>
- The target variable was `Item_Outlet_Sales`. </br>

```python
X = df[['Item_Weight', 'Item_Visibility', 'Item_Type', 'Item_MRP',
        'Outlet_Establishment_Year', 'Outlet_Location_Type',
        'Outlet_Type_Grocery Store', 'Outlet_Type_Supermarket Type1',
        'Outlet_Type_Supermarket Type2', 'Outlet_Type_Supermarket Type3']]

y = df['Item_Outlet_Sales']
```

- Visualized the correlation between features using a heatmap. </br>

```python
corr = df.corr()
sns.heatmap(
    corr,
    vmin=-1,
    vmax=1,
    cmap='coolwarm',
    annot=True,
    fmt='.2f',
    linewidths=0.1
)
plt.show()
```

<!-- Add heatmap image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="850"> </br> -->

- Used `RandomForestRegressor` to calculate feature importance. </br>

```python
model = RandomForestRegressor(random_state=1, max_depth=10)
model.fit(X, y)

features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

- The top important features identified from Random Forest feature importance were: </br>
  - `Item_MRP` </br>
  - `Outlet_Type_Grocery Store` </br>
  - `Outlet_Type_Supermarket Type3` </br>
  - `Item_Visibility` </br>

<!-- Add feature importance output image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="750"> </br> -->

## 7. Modeling </br>

- Trained a Linear Regression model to predict `Item_Outlet_Sales`. </br>
- Split the dataset into training and testing sets using `train_test_split`. </br>

```python
model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    shuffle=True
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 8. Model Evaluation </br>

- Evaluated the Linear Regression model using K-Fold Cross-Validation. </br>
- Used Mean Squared Error (MSE) and R-Squared (R2) as evaluation metrics. </br>

```python
kf = KFold(n_splits=10, shuffle=True, random_state=1)

scoring = {
    'mse': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(r2_score)
}

scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring=scoring,
    return_train_score=True
)

mse_scores = -scores['test_mse']
r2_scores = scores['test_r2']

print(f"Mean Squared Errors for each fold: {mse_scores}")
print(f"Average Mean Squared Error: {mse_scores.mean()}")

print(f"R2 Scores for each fold: {r2_scores}")
print(f"Average R2 Score: {r2_scores.mean()}")
```

- Visualized the distribution of actual and predicted sales values using KDE plots. </br>

```python
plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(y_test, label="Actual Sales")
ax2 = sns.kdeplot(y_pred, label='Predicted Sales', ax=ax1)
plt.legend()
plt.show()
```

<!-- Add evaluation output image here if available -->
<!-- <img src="YOUR_IMAGE_URL" width="850"> </br> -->

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

## 👥 Team Member's Role Division

- Kim Joonhee: Data exploration, data description, histogram visualization, and PPT production </br>
- Kim Sion: Data reduction and feature selection using correlation analysis and Random Forest </br>
- Ju Yongwan: Data preprocessing, Item_Type reclassification, Label Encoding, One-Hot Encoding modification, validation after model learning, and GitHub cleanup </br>
- Kim Hyunjeong: Data value changes, missing-value handling, data dropping, outlier removal, scaling, and encoding </br>

## ✔️ Source

* ChatGPT </br>
* [Big Mart Sales Prediction Dataset](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction) </br>
* [scikit-learn Documentation](https://scikit-learn.org/stable/) </br>
* [Seaborn Documentation](https://seaborn.pydata.org/) </br>
* [Dimensionality Reduction Techniques for Categorical and Continuous Data](https://medium.com/codex/dimensionality-reduction-techniques-for-categorical-continuous-data-75d2bca53100) </br>
* [Regression Model Evaluation Metrics](https://velog.io/@ljs7463/%ED%9A%8C%EA%B7%80%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9Cevaluation-metrics) </br>
* [One-Hot Encoding: pd.get_dummies vs OneHotEncoder](https://coduking.tistory.com/entry/%EC%9B%90-%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9-pdgetdummies-vs-OneHotEncoder) </br>
