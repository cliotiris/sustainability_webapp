import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import shap
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Ensuring that the app utilizes the full width of the browser window
st.set_page_config(
   page_title="Your App Title",
   layout="wide",
   initial_sidebar_state="expanded",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""
    <style>
        .big-font {
            font-size:40px !important;
            font-weight: bold;
            color: #1f77b4;  /* Light green color */
            text-align: center;
            margin-top: -62px;  /* Adjust the negative value as needed */
        }
    </style>
    <p class="big-font">Prediction App for WEEE Operation Management Indicators</p>
    <!-- <p style="text-align:center;">This App predicts the WEEE Operation Management Indicators</p> --> 
""", unsafe_allow_html=True)

# The datasets
data = {
    "Products put on the market": [153058, 175935, 212228, 210356, 215320, 178260, 152318, 135264, 124703, 139454, 125136, 130438, 134420, 145828, 166618, 171792],
    "Waste collected": [763, 11342, 31406, 47142, 66106, 46527, 42360, 37235, 38268, 45420, 49008, 53715, 55831, 58040, 64730, 60863],
    "Recovery": [239, 9365, 24236, 39045, 55883, 45598, 38495, 33578, 35861, 38144, 43888, 49583, 46981, 50425, 57113, 54263],
    "Recycling and preparing for reuse": [239, 9365, 24236, 39045, 55883, 45598, 38495, 33578, 35861, 38144, 43888, 45881, 43998, 46825, 51169, 51476]
}
period = {
    "YEAR": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
}

# Convert the datasets into a DataFrame
df = pd.DataFrame(data)
df['YEAR'] = period['YEAR']

# Step 1: Data Preparation Function
def prepare_data(dataset):
    time = np.array(df['YEAR'])
    series = np.array(df[dataset])
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    split_time = 12
    time_train = time[:split_time]
    x_train = scaled_series[:split_time]
    time_valid = time[split_time:]
    x_valid = scaled_series[split_time:]

    return time_train, x_train, time_valid, x_valid, scaler

# Step 2: Windowed Dataset Function
window_size = 3
batch_size = 8
shuffle_buffer_size = 5

# Windowed_dataset function
@tf.autograph.experimental.do_not_convert
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1][..., np.newaxis], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# Step 3: Build and Compile the CNN-RNN Model Function
def build_model():
    cnn_rnn_model = tf.keras.Sequential([
        # 1D Convolutional layers
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', padding='same',
                               input_shape=[window_size, 1]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        # RNN layers
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.SimpleRNN(16, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])

    # Set the Optimal learning rate
    learning_rate = 8e-4
    # Set the optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    # Set the training parameters
    cnn_rnn_model.compile(loss=tf.keras.losses.Huber(),
                          optimizer=optimizer,
                          metrics=["mae"])
    return cnn_rnn_model

# Step 4: Predict EEE units for future years (2021-2030)
def predict_future(dataset_name, model):
    future_years = np.arange(2021, 2031, 1)
    _, _, _, _, scaler = prepare_data(dataset_name)
    last_data_points = df[dataset_name].values[-window_size:]
    scaled_series_future = scaler.transform(last_data_points.reshape(-1, 1)).flatten()

    future_forecast = []
    for time in range(10):
        prediction = model.predict(scaled_series_future[np.newaxis, :, np.newaxis])
        future_forecast.append(prediction[0, 0])
        scaled_series_future = np.append(scaled_series_future, prediction)
        scaled_series_future = scaled_series_future[-window_size:]

    forecasted_values = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten()
    return future_years, forecasted_values


# Step 5: Plot historical data and predictions
def plot_historical_and_predictions(dataset_name, model):
    future_years, forecasted_values = predict_future(dataset_name, model)

    # Create traces
    trace0 = go.Scatter(x=df['YEAR'], y=df[dataset_name], mode='lines+markers', name='Historical Data')
    trace1 = go.Scatter(x=future_years, y=forecasted_values, mode='lines+markers', name='Predictions')

    layout = go.Layout(
        title=dict(text=f'Historical Data and Predictions for {dataset_name}', x=0.5, xanchor='center', font=dict(size=24)),
        xaxis=dict(title='Year'),
        yaxis=dict(title='Indicators in kilograms'),
        template="plotly_dark",
        font=dict(family="Arial, sans-serif", size=18, color="#555555"),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Create figure
    fig = go.Figure(data=[trace0, trace1], layout=layout)

    # Show figure
    st.plotly_chart(fig, use_container_width=True)

# Step 6: Add a function to display the predictions in a well-shaped table
def show_predictions_table(dataset_name, model):
    future_years, forecasted_values = predict_future(dataset_name, model)
    prediction_data = pd.DataFrame({
        'Year': future_years,
        'Predictions': forecasted_values
    })
    return prediction_data

#######################################################################################################################
# Streamlit SIDEBAR for dataset selection
dataset_name = st.sidebar.selectbox('Select a Dataset:', options=list(data.keys()))
# Plot type selection
plot_type = st.sidebar.selectbox('Select a Plot Type:', options=['Lineplot', 'Scatterplot', 'Histogram', 'Boxplot'])
# Display the selected dataset
selected_data = df[['YEAR', dataset_name]]

# Plot the selected dataset based on the plot type
if plot_type == 'Lineplot':
    fig = px.line(
        selected_data, x='YEAR', y=dataset_name,
        title=f'{dataset_name}',
        line_shape='linear', line_dash_sequence=['solid']
    )
    fig.update_traces(line=dict(width=3))

elif plot_type == 'Scatterplot':
    fig = px.scatter(
        selected_data, x='YEAR', y=dataset_name,
        title=f'{dataset_name}',
        size_max=15, symbol_sequence=['circle']
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))

elif plot_type == 'Histogram':
    fig = px.histogram(
        selected_data,
        x=dataset_name,
        title=f'Distribution of {dataset_name}',
        nbins=20,  # Adjust number of bins according to your preference
        color_discrete_sequence=['skyblue']  # Change color as per preference
    )
    fig.update_traces(
        marker_line_width=1,  # Width of the bin edges
        marker_line_color='white'  # Color of the bin edges
    )
    fig.update_layout(
        bargap=0.1,  # Gap between bars
        xaxis_title='',
        yaxis_title='Frequency',
        xaxis=dict(showticklabels=False)
    )

elif plot_type == 'Boxplot':
    fig = px.box(
        selected_data,
        y=dataset_name,
        title=f'Boxplot of {dataset_name}',
        boxmode='overlay',
        notched=True
    )
    fig.update_traces(
        marker=dict(outliercolor='rgba(219, 64, 82, 0.6)', size=8, line=dict(outlierwidth=2)),
        line=dict(width=2)
    )
    fig.update_layout(
        xaxis_title='',
        yaxis_title=dataset_name,
        xaxis=dict(showticklabels=False)
    )

# Update the layout for a better appearance
layout_settings = {
    'template': "plotly_dark",  # Using a dark theme for a different look
    'font': dict(family="Arial, sans-serif", size=18, color="#555555"),
    'title': dict(x=0.5, xanchor='center', font=dict(size=24, family="Arial, sans-serif", color="#555555"), y=1),
    'margin': dict(l=20, r=20, t=40, b=20),
    'height': 300  # Set the height of the plot here
}

# Apply different settings for Histogram
if plot_type == 'Histogram':
    fig.update_layout(**layout_settings)
else:
    fig.update_layout(**layout_settings, yaxis_title='Indicators in kilograms')

st.plotly_chart(fig, use_container_width=True)

########################################################################################################################
# Main Panel
with st.container():
    # Sidebar
    with st.sidebar:
        show_predictions_button = st.button('Plot and Show the Predictions')

    if show_predictions_button:
        # Check if the model is saved before showing predictions
        if os.path.exists('trained_model'):
            with st.spinner('Generating Predictions...'):
                # Load the trained model
                model = tf.keras.models.load_model('trained_model')

                # Plot the Predictions
                plot_historical_and_predictions(dataset_name, model)

                # Generate and Show the Predictions Table
                future_years, forecasted_values = predict_future(dataset_name, model)
                future_years = [int(year) for year in future_years]  # Convert future_years to integers

                prediction_df = pd.DataFrame({
                    'Year': future_years,
                    'Predictions': np.round(forecasted_values, 0)  # Round 'Predictions' values to the nearest integer
                })

                prediction_df['Predictions'] = prediction_df['Predictions'].apply(
                    lambda x: format(x, ".0f"))  # Remove trailing zeros
                prediction_df = prediction_df.set_index('Year')  # Set 'Year' as the index
                prediction_df.index = prediction_df.index.astype(int)  # Ensure the index is formatted as integers

                # Show the Predictions in a Table
                st.sidebar.header("Predictions:")
                st.sidebar.table(prediction_df)  # Use table instead of write for better formatting

            st.success('Predictions Generated and Displayed!')
