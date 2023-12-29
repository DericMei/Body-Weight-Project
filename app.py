import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
from lstm import LSTMModel

# Load the saved model
model = LSTMModel(1, 32, num_layers=1, dropout_prob=0)
model.load_state_dict(torch.load('model_lstm.pth'))
model.eval()

# Load the saved scaler
scaler = joblib.load('weight_scaler.pkl')

# Function to create sequences
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x.reshape(-1, 1))  # Make sure it's 3D
        ys.append(y.reshape(-1,))
    return np.array(xs), np.array(ys)

# Function to make predictions
def predict_weight(weights):
    # Normalize the input
    weights = scaler.transform(np.array(weights).reshape(-1, 1))
    
    # Create a sequence
    sequence = np.array(weights).reshape(1, -1, 1)  # Reshape to the format the model expects
    
    # Convert to tensor
    sequence = torch.tensor(sequence, dtype=torch.float32)
    
    # Get prediction
    with torch.no_grad():
        prediction = model(sequence).numpy()
    
    # Denormalize the prediction
    prediction = scaler.inverse_transform(prediction)
    
    return prediction.flatten()[0]

# Function to create sequence
def fill_missing_values_with_interpolation(df, date_column='date', value_column='weight'):

    df_new = df.copy()

    # Ensure that 'date' is a datetime column
    df_new[date_column] = pd.to_datetime(df_new[date_column])
    
    # Drop duplicates, keeping only the first entry for each date
    df_new = df_new.drop_duplicates(subset=date_column)
    
    # Set 'date' as the DataFrame index
    df_new.set_index(date_column, inplace=True)
    
    # Create a continuous date range from the start date to the end date
    full_date_range = pd.date_range(start=df_new.index.min(), end=df_new.index.max(), freq='D')
    
    # Reindex the DataFrame with the full date range
    df_new = df_new.reindex(full_date_range)
    
    # Interpolate missing 'weight' values using a time-weighted method
    df_new[value_column] = df_new[value_column].interpolate(method='time')
    
    # Reset the index to turn the 'date' back into a column
    df_new.reset_index(inplace=True)
    
    # Rename the index back to 'date'
    df_new.rename(columns={'index': date_column}, inplace=True)
    
    return df_new

# Function to create dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

########### Prepare the data to make the interactive visualization ##############
df = pd.read_csv('My Body Weight.csv')
df = df[['startDate', 'value']]
df = df.rename(columns={'startDate': 'date', 'value': 'weight'})
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date
df = df.drop(df.index[0:2])
df = df[df['weight'] > 50]
df_final = fill_missing_values_with_interpolation(df)
data = df_final['weight'].values.reshape(-1, 1)
normalized_data = scaler.fit_transform(data)
df_final['weight_normalized'] = normalized_data
bw = df_final['weight_normalized'].values
X, y = create_sequences(bw, 7)
train_size = 0.8
val_size = 0.1
test_size = 0.1
train_idx = int(len(X) * train_size)
val_idx = int(len(X) * (train_size + val_size))
X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
all_dataset = TimeSeriesDataset(X, y)
batch_size = 32
all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
all_predictions = []
with torch.no_grad():
    for sequences, _ in all_loader:
        predictions = model(sequences)
        all_predictions.extend(predictions.cpu().numpy())
all_predictions_original = scaler.inverse_transform(all_predictions)

# 1. Display the interactive visualization
st.title('Eric Mei Weight Prediction App')

st.write('### Interactive Weight Predictions Over Time')

# Create a timeline for the entire dataset
timeline = pd.date_range(start='2018-04-26', periods=len(all_predictions_original), freq='D')

original_weights = df_final['weight'].values

# Create traces with color coding
trace_train = go.Scatter(x=timeline[:len(X_train)], y=all_predictions_original[:len(X_train)].flatten(), mode='lines', name='Training', line=dict(color='green'))
trace_val = go.Scatter(x=timeline[len(X_train):len(X_train)+len(X_val)], y=all_predictions_original[len(X_train):len(X_train)+len(X_val)].flatten(), mode='lines', name='Validation', line=dict(color='orange'))
trace_test = go.Scatter(x=timeline[-len(X_test):], y=all_predictions_original[-len(X_test):].flatten(), mode='lines', name='Testing', line=dict(color='red'))
trace_original = go.Scatter(x=timeline, y=original_weights, mode='lines', name='Actual', line=dict(color='yellow'))

# Create the figure
fig = go.Figure(data=[trace_train, trace_val, trace_test, trace_original])

# Add titles and labels
fig.update_layout(title='Predictions vs. Original Data',
                  xaxis_title='Date',
                  yaxis_title='Weight (kg)',
                  hovermode="x")

# Add interactive slider and buttons
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

st.plotly_chart(fig)

# 2. Predict the next day's weight based on input
st.write('### Predict the Next Day\'s Weight')

# Input fields to enter the 7 days of weight data
input_weights = []
for i in range(1, 8):
    weight = st.number_input(f'Enter weight for Day {i}', min_value=75, value=80)  # You can adjust default and range
    input_weights.append(weight)

if st.button('Predict Next Day\'s Weight'):
    if len(input_weights) == 7:
        # Make prediction
        next_day_weight = predict_weight(input_weights)
        st.write(f'The predicted weight for the next day is: {next_day_weight:.2f} kg')
    else:
        st.error('Please enter 7 days of weights.')
