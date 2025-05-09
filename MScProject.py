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
import xgboost as xgb
import time

column_names = ["Unit No.", "time, in cycles","operational setting 1","operational setting 2","operational setting 3","sensor measurement 1","sensor measurement 2","sensor measurement 3","sensor measurement 4","sensor measurement 5","sensor measurement 6","sensor measurement 7","sensor measurement 8","sensor measurement 9","sensor measurement 10","sensor measurement 11","sensor measurement 12","sensor measurement 13","sensor measurement 14","sensor measurement 15","sensor measurement 16","sensor measurement 17","sensor measurement 18","sensor measurement 19","sensor measurement 20","sensor measurement 21","sensor measurement 22","sensor measurement 23","sensor measurement 24","sensor measurement 25","sensor measurement 26"]
df_train_FD1 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/train_FD001.txt", sep=" ", header=None, names = column_names, engine="python")
df_train_FD2 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/train_FD002.txt", sep=" ", header=None, names = column_names, engine="python")
df_train_FD3 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/train_FD003.txt", sep=" ", header=None, names = column_names, engine="python")
df_train_FD4 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/train_FD004.txt", sep=" ", header=None, names = column_names, engine="python")

df_test_FD1 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/test_FD001.txt", sep=" ", header=None, names = column_names, engine="python")
df_test_FD2 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/test_FD002.txt", sep=" ", header=None, names = column_names, engine="python")
df_test_FD3 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/test_FD003.txt", sep=" ", header=None, names = column_names, engine="python")
df_test_FD4 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/test_FD004.txt", sep=" ", header=None, names = column_names, engine="python")

column_name = ["Actual RUL"]
df_rul_FD1 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/RUL_FD001.txt", header=None, names = column_name, engine="python")
df_rul_FD2 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/RUL_FD002.txt", header=None, names = column_name, engine="python")
df_rul_FD3 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/RUL_FD003.txt", header=None, names = column_name, engine="python")
df_rul_FD4 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/RUL_FD004.txt", header=None, names = column_name, engine="python")


print(df_train_FD1.isnull().sum())
print(df_train_FD2.isnull().sum())
print(df_train_FD3.isnull().sum())
print(df_train_FD4.isnull().sum())
print(df_test_FD1.isnull().sum())
print(df_test_FD2.isnull().sum())
print(df_test_FD3.isnull().sum())
print(df_test_FD4.isnull().sum())

df_train_FD1 = df_train_FD1.dropna(axis=1)
df_train_FD2 = df_train_FD2.dropna(axis=1)
df_train_FD3 = df_train_FD3.dropna(axis=1)
df_train_FD4 = df_train_FD4.dropna(axis=1)
df_test_FD1 = df_test_FD1.dropna(axis=1)
df_test_FD2 = df_test_FD2.dropna(axis=1)
df_test_FD3 = df_test_FD3.dropna(axis=1)
df_test_FD4 = df_test_FD4.dropna(axis=1)


df_train_FD1_desc=df_train_FD1.describe()
df_train_FD2_desc=df_train_FD2.describe()
df_train_FD3_desc=df_train_FD3.describe()
df_train_FD4_desc=df_train_FD4.describe()
df_test_FD1_desc=df_test_FD1.describe()
df_test_FD2_desc=df_test_FD2.describe()
df_test_FD3_desc=df_test_FD3.describe()
df_test_FD4_desc=df_test_FD4.describe()


#df_train_FD1_desc.to_excel("train_FD1_desc.xlsx", index=True)
#df_train_FD2_desc.to_excel("train_FD2_desc.xlsx", index=True)
#df_train_FD3_desc.to_excel("train_FD3_desc.xlsx", index=True)
#df_train_FD4_desc.to_excel("train_FD4_desc.xlsx", index=True)
#df_test_FD1_desc.to_excel("test_FD1_desc.xlsx", index=True)
#df_test_FD2_desc.to_excel("test_FD2_desc.xlsx", index=True)
#df_test_FD3_desc.to_excel("test_FD3_desc.xlsx", index=True)
#df_test_FD4_desc.to_excel("test_FD4_desc.xlsx", index=True)


#Visualization Trends for Train FD1

#filter data for a few example engine unites

#sensor_cols = [f"sensor measurement {i}" for i in range(1,22)]
#sample_units = df_train_FD1['Unit No.'].unique()[95:] #selecting first 5 unique units
#subset_df_train_FD1 = df_train_FD1[df_train_FD1['Unit No.'].isin(sample_units)]

#understanding the distribution of features
#features = ["Unit No.", "time, in cycles"]
#for feature in features:
#    plt.figure(figsize=(6, 4))
#    sns.histplot(df_train_FD1[feature], kde=True)
#    plt.title(f"Distribution of {feature}")
#    plt.show()


# Creating a line pplot for each sensor measurement
#palette = sns.color_palette("tab10", n_colors=len(sample_units))
#for sensor in sensor_cols:
#    plt.figure(figsize=(12,6))
#    sns.scatterplot(data=subset_df_train_FD1, x="time, in cycles", y=sensor, hue="Unit No.", palette=palette)
#    plt.xlabel("Time in Cycles")
#   plt.ylabel(sensor)
#    plt.title(f"{sensor}Trends Over Time")
#    plt.legend(title="Unit No.")
#    plt.show()

columns_to_normalize = df_train_FD1[["operational setting 1","operational setting 2","operational setting 3","sensor measurement 1","sensor measurement 2","sensor measurement 3","sensor measurement 4","sensor measurement 5","sensor measurement 6","sensor measurement 7","sensor measurement 8","sensor measurement 9","sensor measurement 10","sensor measurement 11","sensor measurement 12","sensor measurement 13","sensor measurement 14","sensor measurement 15","sensor measurement 16","sensor measurement 17","sensor measurement 18","sensor measurement 19","sensor measurement 20","sensor measurement 21"]]
#normalization (normalizing the features)
scaler_minmax = MinMaxScaler()
df_train_FD1_normalized = pd.DataFrame(scaler_minmax.fit_transform(columns_to_normalize))

#standardization (standardizing the features)
scaler_standard = StandardScaler()
df_train_FD1_standardized = pd.DataFrame(scaler_standard.fit_transform(columns_to_normalize))

columns_to_drop = ["operational setting 3","sensor measurement 1","sensor measurement 5","sensor measurement 10","sensor measurement 16","sensor measurement 18","sensor measurement 19"]
df_train_FD1=df_train_FD1.drop(columns=columns_to_drop)
df_test_FD1=df_test_FD1.drop(columns=columns_to_drop)

#defining Target Variable (RUL)
max_cycles = df_train_FD1.groupby('Unit No.')['time, in cycles'].transform('max')
df_train_FD1['RUL'] = max_cycles - df_train_FD1['time, in cycles'].round(0).astype(int)


window_size = 5  # Adjust based on degradation pattern
threshold = 3  # Higher threshold prevents removing valid degradation

for col in df_train_FD1.columns[4:]:  
    rolling_mean = df_train_FD1[col].rolling(window=window_size, center=True).mean()
    rolling_std = df_train_FD1[col].rolling(window=window_size, center=True).std()
    
    z_scores = (df_train_FD1[col] - rolling_mean) / rolling_std
    
    # Replace only sudden, isolated spikes
    df_train_FD1[col] = np.where(np.abs(z_scores) > threshold, rolling_mean, df_train_FD1[col])
for col in df_test_FD1.columns[4:]:  
    rolling_mean = df_test_FD1[col].rolling(window=window_size, center=True).mean()
    rolling_std = df_test_FD1[col].rolling(window=window_size, center=True).std()
    
    z_scores = (df_test_FD1[col] - rolling_mean) / rolling_std
    
    # Replace only sudden, isolated spikes
    df_test_FD1[col] = np.where(np.abs(z_scores) > threshold, rolling_mean, df_test_FD1[col])

#df_last_cycle_test = df_test_FD1.groupby("Unit No.")["time, in cycles"].max().reset_index()
# Compute correlation matrix
correlation_matrix = df_train_FD1.corr()
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Extract correlation of features with the target variable 'RUL'
target_correlation = correlation_matrix["RUL"].sort_values(ascending=False)

# Convert to DataFrame for visualization
corr_df = pd.DataFrame(target_correlation)

# Plot heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation with RUL")
plt.show()

#Exponential Weighted Mean for each sensor
for col in df_train_FD1.columns:
    if 'sensor measurement' in col:
        df_train_FD1[f'{col}_ewma'] = df_train_FD1[col].ewm(span=10, adjust=False).mean()
        df_test_FD1[f'{col}_ewma'] = df_test_FD1[col].ewm(span=10, adjust=False).mean()

#First-order difference for each sensor measurement
for col in df_train_FD1.columns:
    if 'sensor measurement' in col:
        df_train_FD1[f'{col}_diff'] = df_train_FD1[col].diff()
        df_test_FD1[f'{col}_diff'] = df_test_FD1[col].diff()

window_size = 10  # For a moving average window of 10 cycles

# Moving Average for each sensor
for col in df_train_FD1.columns:
   if 'sensor measurement' in col:
       df_train_FD1[f'{col}_rolling_mean'] = df_train_FD1[col].rolling(window=window_size).mean()
       df_test_FD1[f'{col}_rolling_mean'] = df_test_FD1[col].rolling(window=window_size).mean() 

# Rolling Standard Deviation for each sensor
for col in df_train_FD1.columns:
    if 'sensor measurement' in col and '_rolling_mean' not in col:
        df_train_FD1[f'{col}_rolling_std'] = df_train_FD1[col].rolling(window=window_size).std()
        df_test_FD1[f'{col}_rolling_std'] = df_test_FD1[col].rolling(window=window_size).std()

df_train_FD1 = df_train_FD1.bfill().ffill()
df_test_FD1 = df_test_FD1.bfill().ffill()

'''
# Columns to exclude
excluded_columns = ["Unit No", "time, in cycles", "operational setting 1", "operational setting 2", "RUL"]

# Ensure 'RUL' is only excluded from train data (not in test data)
train_features = df_train_FD1.drop(columns=excluded_columns, errors='ignore')
test_features = df_test_FD1.drop(columns=[col for col in excluded_columns if col in df_test_FD1.columns], errors='ignore')

# Instantiate and fit PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(train_features)  # Fit only on training data to avoid leakage

# Transform train and test sets
poly_features_train = poly.transform(train_features)
poly_features_test = poly.transform(test_features)

# Get feature names
poly_feature_names = poly.get_feature_names_out(train_features.columns)

# Convert to DataFrame with the same index
poly_df_train = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=df_train_FD1.index)
poly_df_test = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=df_test_FD1.index)

# Concatenate original DataFrame with polynomial features
df_train_FD1 = pd.concat([df_train_FD1, poly_df_train], axis=1)
df_test_FD1 = pd.concat([df_test_FD1, poly_df_test], axis=1)

'''
operational_settings_columns = ['operational setting 1', 'operational setting 2']
sensor_measurements_columns = [col for col in df_train_FD1.columns if 'sensor measurement' in col]

# Standardize the operational settings
scaler = MinMaxScaler()
df_train_FD1[operational_settings_columns] = scaler.fit_transform(df_train_FD1[operational_settings_columns])
df_test_FD1[operational_settings_columns] = scaler.fit_transform(df_test_FD1[operational_settings_columns])

# Standardize the sensor measurements
df_train_FD1[sensor_measurements_columns] = scaler.fit_transform(df_train_FD1[sensor_measurements_columns])
df_test_FD1[sensor_measurements_columns] = scaler.fit_transform(df_test_FD1[sensor_measurements_columns])


# Now, 'df' has the standardized values for both groups of features


# Get the last recorded cycle for each unit
df_last_cycle_test = df_test_FD1.groupby("Unit No.")["time, in cycles"].max().reset_index()

# Merge to keep only the last cycle per unit
df_test_last_FD1 = df_test_FD1.merge(df_last_cycle_test, on=["Unit No.", "time, in cycles"], how="inner")


correlation_matrix = df_train_FD1.corr()
# Extract correlation of features with the target variable 'RUL'
target_correlation = correlation_matrix["RUL"].sort_values(ascending=False)

# Convert to DataFrame for visualization
corr_df = pd.DataFrame(target_correlation)

# Plot heatmap
plt.figure(figsize=(20, 40))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation with RUL")
plt.show()



# Define features (X) and target variable (y)
features = [col for col in df_train_FD1.columns if col not in ["RUL","operational setting 1", "operational setting 2"]]
X = df_train_FD1[features]
y = df_train_FD1["RUL"]
X_test = df_test_last_FD1[features]

        
# Handle NaN values due to rolling statistics
X = X.bfill().ffill()  # Backward & forward filling
X_test = X_test.bfill().ffill()






# Split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#param_grid = {
#    "n_estimators": [50, 100, 200, 300],
#    "max_depth": [10, 20, 30, None],
#    "min_samples_split": [2, 5, 10],
#    "min_samples_leaf": [1, 2, 4, 5],
#}

#rf_random = RandomizedSearchCV(
#    RandomForestRegressor(random_state=42, n_jobs=-1),
#    param_distributions=param_grid,
#    n_iter=10,
#    cv=3,
#    verbose=2,
#    n_jobs=-1,
#    scoring="neg_root_mean_squared_error"
#)

#rf_random.fit(X_train, y_train)

# Best hyperparameters
#print("Best Parameters:", rf_random.best_params_)

# Predict and evaluate again
#best_rf = rf_random.best_estimator_
#y_pred = best_rf.predict(X_val)
#print(f"Optimized RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.2f}")

'''
best_rf = RandomForestRegressor(
   n_estimators=100,
   min_samples_split=2,
   min_samples_leaf=1,
   max_depth=30,
   random_state=42,
   n_jobs=-1
)
'''
best_rf = xgb.XGBRegressor(objective='reg:squarederror', 
                                 n_estimators=500, 
                                 learning_rate=0.05, 
                                 max_depth=6, 
                                 subsample=0.8, 
                                 colsample_bytree=0.8, 
                                 random_state=42)

'''

#kf = KFold(n_splits=5, shuffle=True, random_state=42)
#cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

#selector = RFECV(best_rf, step=5, cv=5, n_jobs=-1)
#selector.fit(X_train, y_train)
#selected_features = X_train.columns[selector.support_]
#X_train = X_train[selected_features]
#X_test = X_test[selected_features]
'''
start_train = time.time()
#Train the model
best_rf.fit(X_train, y_train)

end_train = time.time()
train_time = end_train - start_train
print(f"Training Time: {train_time:.2f} seconds")


# Get feature importance
importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(20)

# Plot important features

top_features.plot(kind='barh')
plt.title("Top 20 Important Features")
plt.show()

start_pred = time.time()
# Make predictions
y_pred = best_rf.predict(X_val)
y_pred = np.round(y_pred)

# Make predictions for test data
y_test_pred = best_rf.predict(X_test)
y_test_pred = np.round(y_test_pred)
end_pred = time.time()
prediction_time = end_pred - start_pred
print(f"Prediction Time: {prediction_time:.2f} seconds")


# Evaluate performance
rmse_optimized = np.sqrt(mean_squared_error(y_val, y_pred))
r2_optimized = r2_score(y_val, y_pred)

print(f"Optimized RMSE: {rmse_optimized:.2f}")
print(f"Optimized R² Score: {r2_optimized:.4f}")
#print(f"Cross-Validation RMSE: {np.sqrt(-cv_scores.mean())}")

df_rul_FD1["Unit No."] = df_test_last_FD1["Unit No."].unique()
df_results = pd.DataFrame({"Unit No.": df_test_last_FD1["Unit No."],"Predicted_RUL": y_test_pred})
df_results = df_results.merge(df_rul_FD1, on="Unit No.")

rmse_test = np.sqrt(mean_squared_error(df_results["Actual RUL"], df_results["Predicted_RUL"]))
r2_test = r2_score(df_results["Actual RUL"], df_results["Predicted_RUL"])

print(f"Tested RMSE: {rmse_test:.2f}")
print(f"Tested R² Score: {r2_test:.4f}")

'''
df_results = pd.DataFrame({"Unit No.": df_test_FD1["Unit No."],"time, in cycles": df_test_FD1["time, in cycles"],"Predicted_RUL": y_test_pred})

# Merge to keep only the last cycle per unit
df_test_last_FD1 = df_results.merge(df_last_cycle_test, on=["Unit No.", "time, in cycles"], how="inner")

df_rul_FD1["Unit No."] = df_test_last_FD1["Unit No."].unique()
df_test_last_FD1 = df_test_last_FD1.merge(df_rul_FD1, on="Unit No.")

rmse_test = np.sqrt(mean_squared_error(df_test_last_FD1["Actual RUL"], df_test_last_FD1["Predicted_RUL"]))
r2_test = r2_score(df_test_last_FD1["Actual RUL"], df_test_last_FD1["Predicted_RUL"])

print(f"Tested RMSE: {rmse_test:.2f}")
print(f"Tested R² Score: {r2_test:.4f}")


Optimized RMSE: 49.94
Optimized R² Score: 0.5970
Tested RMSE: 28.29
Tested R² Score: 0.5365

'''

# Scatter plot of Actual RUL vs. Predicted RUL
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









