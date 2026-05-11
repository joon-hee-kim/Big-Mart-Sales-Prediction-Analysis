# Big-Mart-Sales-Prediction-Analysis

This is the result of an outlet-level sales prediction project using the Big Mart Sales Prediction dataset and machine learning-based data science methods.

## Notice! ##
Go to the following site and download `Train.csv`, then run the code with it.  
Please update the file path in the Python files according to your local environment.

* [Big Mart Sales Prediction Dataset](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction) </br>

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
