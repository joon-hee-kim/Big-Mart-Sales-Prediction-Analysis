import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, cross_validate

# Load data file
df = pd.read_csv('/Users/hurki/Documents/schDataSci/dsTermProject/Train.csv')
print(df.head())
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Check data information
print(f"Print data information:\n{df.info()}")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Fill missing data
print("Replace missing data")
df = df.fillna({'Item_Weight':df['Item_Weight'].mean()})
df = df.fillna({'Outlet_Size':'Medium'})
print(f"Check replace missing data:{df.isnull().sum().sum()}")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Drop unnecessary data
print("Drop unnecessary features")
drop_groups = ['Item_Identifier', 'Item_Fat_Content','Outlet_Identifier', 
               'Outlet_Size']
df = df.drop(columns=drop_groups)
print(df.head())
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Categorize 'Item_Type'
print(f"Print Item types:\n{df['Item_Type'].value_counts()}")
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

print("\nCategorize Item_Type:")
df['Item_Type'] = df['Item_Type'].apply(lambda x : get_category(x))
print(f"New dataframe:\n{df.head()}")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Pre-OneHotEncoding to prevent making new missing values
print("OneHotEncoding for Outlet_Type:")
label = df['Outlet_Type']
label.unique()

ohe = OneHotEncoder(sparse_output=False)
outlet_type_encoded = ohe.fit_transform(label.values.reshape(-1,1))
outlet_type_encoded_df = pd.DataFrame(outlet_type_encoded, columns=ohe.get_feature_names_out(['Outlet_Type']))
df = pd.concat([df, outlet_type_encoded_df], axis=1)
df = df.drop(columns=['Outlet_Type'])

print(f"Result of OneHotEncoding{df.head()}")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Delete Outliers
print("Delete Outliers:")

# Draw numerical columns' boxplot for checking Outlier
whis_value=2.0
plt.figure(figsize=(10, 7))
for i, column in enumerate(['Item_Weight', 'Outlet_Establishment_Year', 'Item_MRP', 'Item_Outlet_Sales'], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(df[column], whis=whis_value)
    plt.title(column)
plt.tight_layout()
plt.show()

# Get and drop 'Item_Outlet_Sales' column's Outlier
column = 'Item_Outlet_Sales'
Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - whis_value * IQR, Q3 + whis_value * IQR
outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
df.drop(outliers.index, inplace=True)

# Check if dropping is well done
print("Result of delete:")
plt.figure(figsize=(5, 5))
sns.boxplot(df[column], whis=whis_value)
plt.title(column)
plt.tight_layout()
plt.show()
print(f"Number of rows removed for {column}:", len(outliers))
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Data preprocessing(Encoding)
print("Encoding datas:")

# Label Encoding
le = LabelEncoder()
df['Item_Type'] = le.fit_transform(df['Item_Type'])

# Ordinal Encoding
oe = OrdinalEncoder()
df['Outlet_Location_Type'] = oe.fit_transform(df[['Outlet_Location_Type']])

print(f"Encoding result:\n{df.head()}")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Data preprocessing(Scaling)
print("Scaling datas:")

# Roburst Scaling
for i in ['Item_Weight']:
    df['Item_Weight'] = RobustScaler().fit_transform(df[['Item_Weight']])

# MinMax Scaling
for i in ['Outlet_Establishment_Year']:
    df['Outlet_Establishment_Year'] = MinMaxScaler().fit_transform(df[['Outlet_Establishment_Year']])

# Standard Scaling
for i in ['Item_MRP']:
    df['Item_MRP'] = StandardScaler().fit_transform(df[['Item_MRP']])
print(f"Scaling result:\n{df.head()}")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Feature selection
print("Devide dataframe into datas and target:")

X = df[['Item_Weight', 'Item_Visibility', 'Item_Type', 'Item_MRP',
        'Outlet_Establishment_Year', 'Outlet_Location_Type',
        'Outlet_Type_Grocery Store', 'Outlet_Type_Supermarket Type1',
        'Outlet_Type_Supermarket Type2', 'Outlet_Type_Supermarket Type3']]
y = df['Item_Outlet_Sales']

# Visualize heat map to see correlation
corr = df.corr()
sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.1)
plt.show()

# Random Forest
model = RandomForestRegressor(random_state=1, max_depth=10)
model.fit(X,y)
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Choose important features to train
important_features = ['Item_MRP', 'Outlet_Type_Grocery Store',
                      'Outlet_Type_Supermarket Type3', 'Item_Visibility']
print(f"Top {len(important_features)} of important features:\n{important_features}")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

# Evaluate
# Linear Regression with K-Fold Cross-Validation
print("Linear Regression with K-Fold Cross-Validation:")
kf = KFold(n_splits=10, shuffle=True, random_state=1)
model = LinearRegression()

# Devide Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, shuffle=True)

# Train data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Define scoring metrics
scoring = {'mse': make_scorer(mean_squared_error, greater_is_better=False),
           'r2': make_scorer(r2_score)}

# Perform cross-validation
scores = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, return_train_score=True)

# Convert scores to positive mean squared error
mse_scores = -scores['test_mse']  
r2_scores = scores['test_r2']

print(f"Mean Squared Errors for each fold: {mse_scores}\n")
print(f"Average Mean Squared Error: {mse_scores.mean()}\n")

print(f"R2 Scores for each fold: {r2_scores}\n")
print(f"Average R2 Score: {r2_scores.mean()}\n")

# Visualize
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label="Actual Sales")
ax2 = sns.kdeplot(y_pred, label='Predicted Sales', ax=ax1)
plt.legend()
plt.show()