# README for `study kaggle house price prediction.ipynb`

## Overview
This Jupyter Notebook (`study kaggle house price prediction.ipynb`) is a technical test script that focuses on data preparation, exploration, analysis, and visualization of a real estate dataset. The dataset used is the **"USA Real Estate Dataset,"** which contains information about properties listed for sale in the USA. The notebook is divided into four main parts:

1. **Data Preparation and Exploration**: This section involves importing necessary libraries, downloading the dataset, handling missing data, and performing initial data exploration.
2. **Data Analysis and Visualization**: This section focuses on calculating key metrics, such as price per square meter, and creating visualizations to understand the distribution of prices and average prices by state.
3. **Basic Price Prediction Model (Python, Machine Learning)**: This section involves building a basic machine learning model to predict property prices based on the dataset. Linear Regression is used in this section.
4. **Bonus (if time permits)**: This section includes experimenting with different machine learning models, such as Linear Regression and Random Forest, and performing more advanced data visualizations or exploratory data analysis.

---

## Dataset
The dataset includes information such as:
- `brokered_by`: The broker handling the property.
- `status`: The status of the property (e.g., for sale).
- `price`: The price of the property.
- `bed`: Number of bedrooms.
- `bath`: Number of bathrooms.
- `acre_lot`: Size of the lot in acres.
- `street`: Street address.
- `city`: City where the property is located.
- `state`: State where the property is located.
- `zip_code`: ZIP code of the property.
- `house_size`: Size of the house in square feet.
- `prev_sold_date`: Date when the property was previously sold.

The dataset is downloaded from Kaggle using the Kaggle API.

---

## Key Steps in the Notebook

### 1. Data Preparation and Exploration
- **Importing Libraries**: Libraries such as `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` are imported.
- **Downloading the Dataset**: The dataset is downloaded using the `kaggle datasets download` command.
- **Loading the Dataset**: The dataset is extracted from the ZIP file and loaded into a Pandas DataFrame.
- **Handling Missing Data**: Missing values in key columns (`price`, `state`, `house_size`) are handled by dropping rows with missing values.
- **Initial Data Exploration**: The first few rows of the dataset are displayed for initial understanding.

### 2. Data Analysis and Visualization
- **Calculating Price per Square Meter**: A new column `price_per_sqm` is calculated by dividing the `price` by the `house_size`.
- **Distribution of Price per Square Meter**: A histogram is plotted to visualize the distribution of the price per square meter, using a logarithmic scale.
- **Average Price per Square Meter by State**: The average price per square meter is calculated for each state, and a horizontal bar plot is created to visualize the results.

### 3. Basic Price Prediction Model (Python, Machine Learning)
- **Feature Selection**: Relevant features for predicting property prices are selected.
- **Data Preprocessing**: The data is preprocessed, including handling categorical variables (e.g., `state`, `city`) using one-hot encoding and splitting the data into training and testing sets.
- **Model Training**: A basic machine learning model (Linear Regression) is trained on the training data.
- **Model Evaluation**: The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and R-squared (R²) on the test data.

### 4. Bonus (if time permits)
- **Experimenting with Different Models**: Different machine learning models (Linear Regression and Random Forest) are experimented with, and their performance is compared.
- **Advanced Data Visualizations**: More advanced visualizations, such as pair plots, correlation heatmaps, or interactive plots using `plotly`, are created.
- **Exploratory Data Analysis (EDA)**: Further exploratory data analysis is performed to uncover additional patterns or relationships in the data.

---

## Visualizations
The notebook includes the following visualizations:
- **Histogram of Price per Square Meter**: Shows the distribution of the price per square meter with a logarithmic scale.
- **Bar Plot of Average Price per Square Meter by State**: Shows the average price per square meter for each state.

---

## Requirements
To run this notebook, you will need the following Python libraries installed:
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `numpy`
- `kaggle` (for downloading the dataset)

You can install these libraries using `pip`:
```bash
pip install pandas matplotlib seaborn scikit-learn numpy plotly kaggle
