import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import time


# Load Data (Same as your original script)
column_names = ["Unit No.", "time, in cycles","operational setting 1","operational setting 2","operational setting 3"] + \
               [f"sensor measurement {i}" for i in range(1, 27)]

df_train_FD3 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/train_FD003.txt", sep=" ", header=None, names=column_names, engine="python")
df_test_FD3 = pd.read_csv("C:/Users/raviv/.spyder-py3/CMAPSSData/test_FD003.txt", sep=" ", header=None, names=column_names, engine="python")

df_rul_FD3 = pd.read_excel("C:/Users/raviv/.spyder-py3/CMAPSSData/RUL-03.xlsx")

# Drop NaN columns
df_train_FD3.dropna(axis=1, inplace=True)
df_test_FD3.dropna(axis=1, inplace=True)
columns_to_drop = ["operational setting 3","sensor measurement 1","sensor measurement 5","sensor measurement 10","sensor measurement 16","sensor measurement 18","sensor measurement 19"]
df_train_FD3=df_train_FD3.drop(columns=columns_to_drop)
df_test_FD3=df_test_FD3.drop(columns=columns_to_drop)

# Define Target Variable (RUL)
max_cycles = df_train_FD3.groupby('Unit No.')['time, in cycles'].transform('max')
df_train_FD3['RUL'] = max_cycles - df_train_FD3['time, in cycles'].round(0).astype(int)


#df_train_FD2.to_excel("df_train_FD2.xlsx", index = True)
correlation_matrix = df_train_FD3.corr()
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# Extract correlation of features with the target variable 'RUL'
target_correlation = correlation_matrix["RUL"].sort_values(ascending=False)

# Convert to DataFrame for visualization
corr_df = pd.DataFrame(target_correlation)

max_cycles_test = df_test_FD3.groupby('Unit No.')['time, in cycles'].transform('max')
max_cycles_per_unit= max_cycles_test.to_frame()
max_cycles_per_unit["Unit No."]=df_test_FD3["Unit No."]
df_FD3_merged = max_cycles_per_unit.merge(df_rul_FD3, on='Unit No.', how='left')
df_FD3_merged['Actual Max Cycles'] = df_FD3_merged['time, in cycles'] + df_FD3_merged['Actual RUL']
df_test_FD3['RUL']= df_FD3_merged["Actual Max Cycles"] - df_test_FD3["time, in cycles"].round(0).astype(int)
#df_last_cycle_test = df_test_FD2.groupby("Unit No.")["time, in cycles"].max().reset_index()


#Exponential Weighted Mean for each sensor
for col in df_train_FD3.columns:
    if 'sensor measurement' in col:
        df_train_FD3[f'{col}_ewma'] = df_train_FD3[col].ewm(span=10, adjust=False).mean()
        df_test_FD3[f'{col}_ewma'] = df_test_FD3[col].ewm(span=10, adjust=False).mean()

#First-order difference for each sensor measurement
for col in df_train_FD3.columns:
    if 'sensor measurement' in col:
        df_train_FD3[f'{col}_diff'] = df_train_FD3[col].diff()
        df_test_FD3[f'{col}_diff'] = df_test_FD3[col].diff()

window_size = 10  # For a moving average window of 10 cycles

# Moving Average for each sensor
for col in df_train_FD3.columns:
   if 'sensor measurement' in col:
       df_train_FD3[f'{col}_rolling_mean'] = df_train_FD3[col].rolling(window=window_size).mean()
       df_test_FD3[f'{col}_rolling_mean'] = df_test_FD3[col].rolling(window=window_size).mean() 

# Rolling Standard Deviation for each sensor
for col in df_train_FD3.columns:
    if 'sensor measurement' in col and '_rolling_mean' not in col:
        df_train_FD3[f'{col}_rolling_std'] = df_train_FD3[col].rolling(window=window_size).std()
        df_test_FD3[f'{col}_rolling_std'] = df_test_FD3[col].rolling(window=window_size).std()


df_train_FD3 = df_train_FD3.bfill().ffill()
df_test_FD3 = df_test_FD3.bfill().ffill()

operational_settings_columns = ['operational setting 1', 'operational setting 2']
sensor_measurements_columns = [col for col in df_train_FD3.columns if 'sensor measurement' in col]

# Standardize the operational settings
scaler = MinMaxScaler()
df_train_FD3[operational_settings_columns] = scaler.fit_transform(df_train_FD3[operational_settings_columns])
df_test_FD3[operational_settings_columns] = scaler.fit_transform(df_test_FD3[operational_settings_columns])

# Standardize the sensor measurements
df_train_FD3[sensor_measurements_columns] = scaler.fit_transform(df_train_FD3[sensor_measurements_columns])
df_test_FD3[sensor_measurements_columns] = scaler.fit_transform(df_test_FD3[sensor_measurements_columns])

'''
# Get the last recorded cycle for each unit in the test set
df_last_cycle_test = df_test_FD2.groupby("Unit No.")["time, in cycles"].max().reset_index()
df_test_last_FD2 = df_test_FD2.merge(df_last_cycle_test, on=["Unit No.", "time, in cycles"], how="inner")
df_rul_FD2["Unit No."] = df_test_last_FD2.merge(df_rul_FD2, on= ["Unit No."])

'''
#df_last_cycle_test.to_excel("df2.xlsx", index=True)
#df_test_last_FD2.to_excel("df2_test_last_desc.xlsx", index=True)

#df_test_last = df_test_FD2.merge(df_last_cycle_test, on=["Unit No.", "time, in cycles"], how = "inner")
# Select Features
features = [col for col in df_train_FD3.columns if col not in ["RUL","operational setting 1","operational setting 2"]]
X = df_train_FD3[features]
y = df_train_FD3["RUL"]

X_test = df_test_FD3[features]
#X_test_last = df_test_last[features]
y_test = df_test_FD3["RUL"]
#y_actual_rul = df_rul_FD2["Actual RUL"]


# Convert Data to 3D for LSTM (Sliding Window)
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 150  # Use last 30 cycles for prediction


X_train_seq, y_train_seq = create_sequences(X, y, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
#X_test_last_seq, y_actual_rul_seq = create_sequences(X_test_last, y_actual_rul, time_steps) 

# Split into Train & Validation
X_train, X_val, y_train, y_val = train_test_split(X_train_seq, y_train_seq, test_size=0.1, random_state=42, shuffle=False)
'''
model = Sequential([
    LSTM(32, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01),input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(16, activation='tanh',kernel_regularizer=l2(0.01),return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(8, activation='tanh', return_sequences=False,kernel_regularizer=l2(0.01)),  # Last LSTM layer
    BatchNormalization(),
    Dropout(0.3),

    Dense(16, activation="relu",kernel_regularizer=l2(0.01)),
    Dense(1)  # Output layer for RUL prediction
])
'''
model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.005), input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(32, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation="relu", kernel_regularizer=l2(0.005)),
    Dense(32, activation="relu"),
    Dense(1)
])
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * (0.95 ** epoch))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)


#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4), loss=tf.keras.losses.LogCosh(), metrics=['mae'])

# Train Model
start_train = time.time()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32,callbacks=[lr_scheduler, early_stopping,reduce_lr], verbose=1)
end_train = time.time()
train_time = end_train - start_train
print(f"Training Time: {train_time:.2f} seconds")


start_pred = time.time()
# Make Predictions
y_pred = model.predict(X_val)
y_pred = np.round(y_pred)
y_test_pred = model.predict(X_test_seq)
#y_pred_last = model.predict(X_test_last_seq)
y_test_pred = np.round(y_test_pred)
end_pred = time.time()
prediction_time = end_pred - start_pred
print(f"Prediction Time: {prediction_time:.2f} seconds")

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], 'bo-', label='Training Loss')  # Blue line with circles
plt.plot(history.history['val_loss'], 'r*-', label='Validation Loss')  # Red line with stars

# Labels, title, and grid
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Evaluate Performance
rmse_optimized = np.sqrt(mean_squared_error(y_val, y_pred))
r2_optimized = r2_score(y_val, y_pred)

print(f"Optimized RMSE: {rmse_optimized:.2f}")
print(f"Optimized R² Score: {r2_optimized:.4f}")

rmse_test = np.sqrt(mean_squared_error(y_test_seq, y_test_pred))
r2_test = r2_score(y_test_seq, y_test_pred)

print(f"Tested RMSE: {rmse_test:.2f}")
print(f"Tested R² Score: {r2_test:.4f}")


# Define the grouping size (every 100 points)
group_size = 100

# Compute averages every 100 points
y_test_seq_avg = np.array([np.mean(y_test_seq[i:i+group_size]) for i in range(0, len(y_test_seq), group_size)])
y_test_pred_avg = np.array([np.mean(y_test_pred[i:i+group_size]) for i in range(0, len(y_test_pred), group_size)])

plt.figure(figsize=(8, 6))

# Scatter plot with reduced data points
plt.scatter(y_test_seq_avg, y_test_pred_avg, color='red', label="Predicted RUL (Averaged)", alpha=0.7, marker="x", s=40)
plt.scatter(y_test_seq_avg, y_test_seq_avg, color='blue', label="Actual RUL (Averaged)", alpha=0.7, marker="o", s=40)

# Add a reference y=x line
min_val = min(y_test_seq_avg.min(), y_test_pred_avg.min())
max_val = max(y_test_seq_avg.max(), y_test_pred_avg.max())
plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1, label="Ideal Fit")

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Averaged Predicted vs Actual RUL")
plt.legend()
plt.grid(True)

plt.show()

'''
rmse_last = np.sqrt(mean_squared_error(y_actual_rul_seq, y_pred_last))
r2_last = r2_score(y_actual_rul_seq, y_pred_last)


print(f"RMSE for Last Cycle: {rmse_last:.2f}")
print(f"R² Score for Last Cycle: {r2_last:.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(y_actual_rul_seq, y_pred_last, color='red', label="Predicted RUL", alpha=0.6, marker="x")
plt.plot([y_actual_rul_seq.min(), y_actual_rul_seq.max()],
         [y_actual_rul_seq.min(), y_actual_rul_seq.max()],
         color="black", linestyle="--", linewidth=1, label="Ideal Fit")

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs Actual RUL for Last Cycle")
plt.legend()
plt.grid(True)
plt.show()

'''