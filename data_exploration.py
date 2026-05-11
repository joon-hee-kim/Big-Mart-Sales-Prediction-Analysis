import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder

# Load Data
df = pd.read_csv("/Users/hurki/Documents/schDataSci/dsTermProject/Train.csv")

print("Dataset head 5:\n", df.head(5))

# Print dataset statistical data
print("Dataset shape:", df.shape)
print("Dataset index:", df.index)
print("Dataset columns:", df.columns)
print("\nStatistical Dataset\n", df.describe())
print("\nㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\nFeature Names and Data Types\n", df.dtypes)

# List of columns to be encoded and used for histogram
list_columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Outlet_Sales']

# Ordinal Encoding: 'Item_Fat_Content'
oe = OrdinalEncoder()
df['Item_Fat_Content'] = oe.fit_transform(df[['Item_Fat_Content']])

# Label Encoding: 'Item_Type'
le = LabelEncoder()
df['Item_Type'] = le.fit_transform(df['Item_Type'])

# Ordinal Encoding: 'Outlet_Location_Type', 'Outlet_Size'
df['Outlet_Location_Type'] = oe.fit_transform(df[['Outlet_Location_Type']])
df['Outlet_Size'] = oe.fit_transform(df[['Outlet_Size']])

# OneHotEncoding: 'Outlet_Type'
ohe = OneHotEncoder(sparse_output=False)
outlet_type_encoded = ohe.fit_transform(df[['Outlet_Type']])
outlet_type_encoded_df = pd.DataFrame(outlet_type_encoded, columns=ohe.get_feature_names_out(['Outlet_Type']))
df = pd.concat([df, outlet_type_encoded_df], axis=1)

# Instead of dropping the original 'Outlet_Type' column, we include the encoded columns in the histogram
list_columns.extend(ohe.get_feature_names_out(['Outlet_Type']))
list_columns.remove('Outlet_Type')  # Exclude the original 'Outlet_Type' column

# Histogram
plt.figure(figsize=(15, 8))
df[list_columns].hist(bins=30, figsize=(15, 8))
plt.tight_layout()
plt.show()

