import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import butter, filtfilt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import time


column_names = ["Unit No.", "time in cycles","operational setting 1","operational setting 2","operational setting 3","sensor measurement 1","sensor measurement 2","sensor measurement 3","sensor measurement 4","sensor measurement 5","sensor measurement 6","sensor measurement 7","sensor measurement 8","sensor measurement 9","sensor measurement 10","sensor measurement 11","sensor measurement 12","sensor measurement 13","sensor measurement 14","sensor measurement 15","sensor measurement 16","sensor measurement 17","sensor measurement 18","sensor measurement 19","sensor measurement 20","sensor measurement 21","sensor measurement 22","sensor measurement 23","sensor measurement 24","sensor measurement 25","sensor measurement 26"]

df_train_FD4 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/train_FD004.txt", sep=" ", header=None, names = column_names, engine="python")
df_test_FD4 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/test_FD004.txt", sep=" ", header=None, names = column_names, engine="python")

column_name = ["Actual RUL"]

df_rul_FD4 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/RUL_FD004.txt", header=None, names = column_name, engine="python")

df_train_FD4 = df_train_FD4.dropna(axis=1)
df_test_FD4 = df_test_FD4.dropna(axis=1)

#defining Target Variable (RUL)
max_cycles = df_train_FD4.groupby('Unit No.')['time in cycles'].transform('max')
df_train_FD4['RUL'] = max_cycles - df_train_FD4['time in cycles'].round(0).astype(int)

#df_last_cycle_test = df_test_FD3.groupby("Unit No.")["time, in cycles"].max().reset_index()
# Compute correlation matrix
correlation_matrix = df_train_FD4.corr()
# Plot the heatmap
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Extract correlation of features with the target variable 'RUL'
target_correlation = correlation_matrix["RUL"].sort_values(ascending=False)

# Convert to DataFrame for visualization
corr_df = pd.DataFrame(target_correlation)

# Plot heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation FD3 with RUL")
plt.show()


#Exponential Weighted Mean for each sensor
for col in df_train_FD4.columns:
    if 'sensor measurement' in col:
        df_train_FD4[f'{col}_ewma'] = df_train_FD4[col].ewm(span=10, adjust=False).mean()
        df_test_FD4[f'{col}_ewma'] = df_test_FD4[col].ewm(span=10, adjust=False).mean()

#First-order difference for each sensor measurement
for col in df_train_FD4.columns:
    if 'sensor measurement' in col:
        df_train_FD4[f'{col}_diff'] = df_train_FD4[col].diff()
        df_test_FD4[f'{col}_diff'] = df_test_FD4[col].diff()

window_size = 10  # For a moving average window of 10 cycles

# Moving Average for each sensor
for col in df_train_FD4.columns:
   if 'sensor measurement' in col:
       df_train_FD4[f'{col}_rolling_mean'] = df_train_FD4[col].rolling(window=window_size).mean()
       df_test_FD4[f'{col}_rolling_mean'] = df_test_FD4[col].rolling(window=window_size).mean() 

# Rolling Standard Deviation for each sensor
for col in df_train_FD4.columns:
    if 'sensor measurement' in col and '_rolling_mean' not in col:
        df_train_FD4[f'{col}_rolling_std'] = df_train_FD4[col].rolling(window=window_size).std()
        df_test_FD4[f'{col}_rolling_std'] = df_test_FD4[col].rolling(window=window_size).std()


df_train_FD4 = df_train_FD4.bfill().ffill()
df_test_FD4 = df_test_FD4.bfill().ffill()
# Instantiate PolynomialFeatures to capture interactions (degree 2 for quadratic features)
#poly = PolynomialFeatures(degree=2, include_bias=False)

# Assuming that 'sensor measurement' and 'operational setting' columns are your features
#Combine them into a new dataframe
#features = df_train_FD3[["operational setting 1","operational setting 2","sensor measurement 2","sensor measurement 3","sensor measurement 4","sensor measurement 6","sensor measurement 7","sensor measurement 8","sensor measurement 9","sensor measurement 11","sensor measurement 12","sensor measurement 13","sensor measurement 14","sensor measurement 15","sensor measurement 17","sensor measurement 20","sensor measurement 21"]]
#features1 = df_test_FD3[["operational setting 1","operational setting 2","sensor measurement 2","sensor measurement 3","sensor measurement 4","sensor measurement 6","sensor measurement 7","sensor measurement 8","sensor measurement 9","sensor measurement 11","sensor measurement 12","sensor measurement 13","sensor measurement 14","sensor measurement 15","sensor measurement 17","sensor measurement 20","sensor measurement 21"]]
'''
# Apply Polynomial Features
poly_features = poly.fit_transform(features)
poly_features1 = poly.fit_transform(features1)

#Add the polynomial features to the dataframe
poly_feature_names = poly.get_feature_names_out(features.columns)
poly_feature_names1 = poly.get_feature_names_out(features1.columns)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
poly_df1 = pd.DataFrame(poly_features1,columns=poly_feature_names1)
# Concatenate original dataframe with polynomial features
df_train_FD3 = pd.concat([df_train_FD3, poly_df], axis=1)
df_test_FD3 = pd.concat([df_test_FD3, poly_df1], axis=1)

'''
operational_settings_columns = ['operational setting 1', 'operational setting 2','operational setting 3']
sensor_measurements_columns = [col for col in df_train_FD4.columns if 'sensor measurement' in col]

# Standardize the operational settings
scaler = MinMaxScaler()
df_train_FD4[operational_settings_columns] = scaler.fit_transform(df_train_FD4[operational_settings_columns])
df_test_FD4[operational_settings_columns] = scaler.fit_transform(df_test_FD4[operational_settings_columns])

# Standardize the sensor measurements
df_train_FD4[sensor_measurements_columns] = scaler.fit_transform(df_train_FD4[sensor_measurements_columns])
df_test_FD4[sensor_measurements_columns] = scaler.fit_transform(df_test_FD4[sensor_measurements_columns])


# Now, 'df' has the standardized values for both groups of features
# Get the last recorded cycle for each unit
df_last_cycle_test = df_test_FD4.groupby("Unit No.")["time in cycles"].max().reset_index()

# Merge to keep only the last cycle per unit
df_test_last_FD4 = df_test_FD4.merge(df_last_cycle_test, on=["Unit No.", "time in cycles"], how="inner")

# Compute correlation matrix
correlation_matrix = df_train_FD4.corr()

# Extract correlation of features with the target variable 'RUL'
target_correlation = correlation_matrix["RUL"].sort_values(ascending=False)

# Convert to DataFrame for visualization
corr_df = pd.DataFrame(target_correlation)

# Plot heatmap
plt.figure(figsize=(20, 40))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation FD3 with RUL")
plt.show()

#df_train_FD3.to_excel("train_FD3_scaled.xlsx", index=True)
#df_test_FD3.to_excel("test_FD3_scaled.xlsx", index=True)
# Define features (X) and target variable (y)
features = [col for col in df_train_FD4.columns if col not in ["RUL"]]
X = df_train_FD4[features]
y = df_train_FD4["RUL"]
X_test = df_test_last_FD4[features]

        
# Handle NaN values due to rolling statistics
X = X.bfill().ffill()  # Backward & forward filling
X_test = X_test.bfill().ffill()




# Split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define hyperparameter grid for tuning
#param_grid = {
#    'num_leaves': [31, 50, 100],
#    'learning_rate': [0.01, 0.05, 0.1],
#    'n_estimators': [100, 500, 1000]
#}

# Initialize LightGBM model
lgbm = lgb.LGBMRegressor(learning_rate = 0.05, n_estimators = 1000, num_leaves = 31)

# Perform Grid Search for best hyperparameters
#lgbm_grid = GridSearchCV(lgbm, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
start_train = time.time()
lgbm.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train
print(f"Training Time: {train_time:.2f} seconds")


# Get feature importance
importances = pd.Series(lgbm.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(20)

# Plot important features

top_features.plot(kind='barh')
plt.title("Top 20 Important Features")
plt.show()

# Get the best model and parameters
#best_lgbm = lgbm_grid.best_estimator_
#best_params = lgbm_grid.best_params_
#print(f"Best Parameters: {best_params}")

# Train the best LightGBM model
#best_lgbm.fit(X_train, y_train)

start_pred = time.time()
# Make predictions
y_pred = lgbm.predict(X_val)
y_pred = np.round(y_pred)
y_test_pred = lgbm.predict(X_test)
y_test_pred = np.round(y_test_pred)
end_pred = time.time()
prediction_time = end_pred - start_pred
print(f"Prediction Time: {prediction_time:.2f} seconds")

# Evaluate performance
rmse_optimized = np.sqrt(mean_squared_error(y_val, y_pred))
r2_optimized = r2_score(y_val, y_pred)

print(f"Optimized RMSE: {rmse_optimized:.2f}")
print(f"Optimized R² Score: {r2_optimized:.4f}")

# Prepare results dataframe
df_rul_FD4["Unit No."] = df_test_last_FD4["Unit No."].unique()
df_results = pd.DataFrame({"Unit No.": df_test_last_FD4["Unit No."], "Predicted_RUL": y_test_pred})
df_results = df_results.merge(df_rul_FD4, on="Unit No.")

rmse_test = np.sqrt(mean_squared_error(df_results["Actual RUL"], df_results["Predicted_RUL"]))
r2_test = r2_score(df_results["Actual RUL"], df_results["Predicted_RUL"]) 

print(f"Tested RMSE: {rmse_test:.2f}")
print(f"Tested R² Score: {r2_test:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_results["Actual RUL"], y=df_results["Predicted_RUL"], alpha=0.7, edgecolor="k")

# Reference line (y = x) for perfect predictions
max_rul = max(df_results["Actual RUL"].max(), df_results["Predicted_RUL"].max())
plt.plot([0, max_rul], [0, max_rul], linestyle="--", color="red", label="Perfect Prediction (y = x)")

# Labels and title
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs. Predicted RUL")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()










