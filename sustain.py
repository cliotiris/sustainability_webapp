import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
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
GREECE = {
    "Products put on the market": [153058, 175935, 212228, 210356, 215320, 178260, 152318, 135264, 124703, 139454, 125136, 130438, 134420, 145828, 166618, 171792],
    "Waste collected": [763, 11342, 31406, 47142, 66106, 46527, 42360, 37235, 38268, 45420, 49008, 53715, 55831, 58040, 64730, 60863],
    "Recovery": [239, 9365, 24236, 39045, 55883, 45598, 38495, 33578, 35861, 38144, 43888, 49583, 46981, 50425, 57113, 54263],
    "Recycling and preparing for reuse": [239, 9365, 24236, 39045, 55883, 45598, 38495, 33578, 35861, 38144, 43888, 45881, 43998, 46825, 51169, 51476]
}
GERMANY = {
    "Products put on the market": [1469530, 1836913, 1612228, 1883544, 1660390, 1730794, 1669939, 1776492, 1609232, 1713902, 1897480, 1957989, 2081223, 2375643, 2590244, 2847926],
    "Waste collected": [603120, 753900, 586966, 693775, 832236, 777035, 710250, 690711, 727998, 722968, 721870, 782214, 836907, 853125, 947067, 1037019],
    "Recovery": [546419, 683024, 547364, 643311, 776177, 736321, 673880, 653372, 686940, 689910, 652130, 756964, 811435, 830042, 921577, 1018710],
    "Recycling and preparing for reuse": [480049, 600062, 474436, 558907, 668594, 643079, 595887, 576848, 602894, 608587, 572564, 678272, 717823, 729864, 808441, 899287]
}
NETHERLANDS = {
    "Products put on the market": [49700, 62125, 57218, 78215, 59610, 61696, 65500, 324717, 306011, 319951, 342762, 371592, 417445, 492107, 629363, 735844],
    "Waste collected": [89827, 94484, 98190, 103319, 108457, 128119, 132197, 123684, 117499, 141805, 145192, 154675, 170849, 184426, 198649, 220236],
    "Recovery": [66229, 82787, 88356, 93822, 97385, 121084, 123620, 118036, 112281, 137013, 139034, 148219, 159569, 166192, 161519, 196048],
    "Recycling and preparing for reuse": [58780, 73475, 77617, 83756, 85515, 102325, 108102, 102614, 97669, 116841, 119154, 125065, 139367, 139367, 138189, 162937]
}
FRANCE = {
    "Products put on the market": [1185250, 1481563, 1637531, 1669718, 1565469, 1635493, 1664816, 1602702, 1554732, 1554305, 1676384, 1748258, 1880274, 1928995, 2094364, 2179743],
    "Waste collected": [12128, 15160, 174777, 300988, 393273, 433959, 470192, 470556, 479694, 522793, 617401, 721949, 742333, 814385, 846229, 835141],
    "Recovery": [3663, 4579, 134809, 242298, 311306, 356658, 394691, 394298, 415279, 468118, 552822, 651131, 665775, 674846, 729832, 758643],
    "Recycling and preparing for reuse": [3264, 4081, 125122, 228394, 290526, 335991, 366057, 361127, 377669, 422589, 505466, 591105, 608127, 604571, 647669, 666178]
}
ITALY = {
    "Products put on the market": [894465, 1118082, 1397603, 1391855, 973713, 1117406, 993997, 892910, 846720, 883883, 912349, 1011617, 1026864, 1482787, 1422251, 1560600],
    "Waste collected": [299226, 374033, 467542, 448030, 521113, 582482, 544577, 497378, 437090, 314210, 344629, 361322, 381656, 421234, 461969, 477972],
    "Recovery": [267641, 334552, 418190, 485236, 478463, 525476, 510958, 401730, 389622, 267888, 304358, 321258, 350990, 370237, 394158, 429810],
    "Recycling and preparing for reuse": [208817, 261021, 326277, 381794, 455234, 502292, 506964, 397675, 384578, 258679, 295381, 310736, 339453, 354926, 379812, 412921]
}
period = {
    "YEAR": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
}

# Mapping the countries to their respective datasets
country_datasets = {
    'Greece': GREECE,
    'Germany': GERMANY,
    'Netherlands': NETHERLANDS,
    'France': FRANCE,
    'Italy': ITALY,
}

# Sidebar controls
countries = list(country_datasets.keys())
categories = ['Products put on the market', 'Waste collected','Recovery', 'Recycling and preparing for reuse']
country = st.sidebar.selectbox('Select a Country', countries)
selected_category = st.sidebar.selectbox('Select a Category', categories)

# Updating the dataframe based on the selected country
df = pd.DataFrame(country_datasets[country])
df['YEAR'] = period['YEAR']

# Automatically plot the dataset as a line plot once the user selects a category
selected_data = df[['YEAR', selected_category]]

# Layout settings
layout_settings = {
    'template': "plotly_dark",  # Using a dark theme for a different look
    'font': dict(family="Arial, sans-serif", size=18, color="#555555"),
    'title': dict(x=0.5, xanchor='center', font=dict(size=24, family="Arial, sans-serif", color="#555555"), y=1),
    'margin': dict(l=20, r=20, t=40, b=20),
    'height': 300,  # Set the height of the plot here
    'yaxis': dict(title="tonnes", titlefont=dict(size=18, family="Arial, sans-serif", color="#555555")),
    'xaxis': dict(title="", showgrid=True, gridwidth=1, gridcolor='lightgrey')
}


# Creating the plot with updated layout settings
fig = px.line(selected_data, x='YEAR', y=selected_category, title=f'{selected_category} in {country}')
fig.update_layout(**layout_settings)
st.plotly_chart(fig, use_container_width=True)

# Data preparation steps
# Ensure this is placed before any button conditions to ensure it runs every time
series = np.array(df[selected_category], dtype=np.float32)
time = np.array(period['YEAR'], dtype=np.float32)
# Normalize Data
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Split Data
split_time = 12
time_train = time[:split_time]
x_train_unscaled = series[:split_time]
x_train = scaled_series[:split_time]
time_valid = time[split_time:]
x_valid_unscaled = series[split_time:]
x_valid = scaled_series[split_time:]

# Prepare the Windowed Dataset
window_size = 3
batch_size = 8
shuffle_buffer_size = 5

@tf.autograph.experimental.do_not_convert
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1][..., np.newaxis], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
valid_dataset = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)

# Define the Model
def create_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', padding='same',
                               input_shape=[window_size, 1]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.SimpleRNN(16, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    return model

# Sidebar for user interactions
with st.sidebar:
    # Train the Model Button
    if st.button('Train the Model'):
        with st.spinner('Training the Model...'):
            cnn_rnn_model = create_model(window_size)
            learning_rate = 8e-4
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
            cnn_rnn_model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                                              restore_best_weights=True)
            history = cnn_rnn_model.fit(train_dataset, validation_data=valid_dataset, epochs=300, verbose=1,
                                        callbacks=[early_stopping])
            st.session_state['cnn_rnn_model'] = cnn_rnn_model
            st.session_state['scaler'] = scaler
            st.session_state['scaled_series'] = scaled_series
            st.session_state['window_size'] = window_size
            st.success('Model Training Completed!')

    # Evaluate the Model Button
    if st.sidebar.button('Evaluate the Model'):
        if 'cnn_rnn_model' in st.session_state:
            with st.spinner('Evaluating the Model...'):
                forecast = []
                cnn_rnn_model = st.session_state['cnn_rnn_model']
                scaler = st.session_state['scaler']
                scaled_series = st.session_state['scaled_series']
                window_size = st.session_state['window_size']
                for time in range(len(scaled_series) - window_size):
                    forecast.append(
                        cnn_rnn_model.predict(scaled_series[time:time + window_size][np.newaxis, :, np.newaxis]))
                forecast = forecast[split_time - window_size:]
                unscaled_forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
                results = np.array(unscaled_forecast).squeeze()
                mae = mean_absolute_error(x_valid_unscaled, results)
                rmse = np.sqrt(mean_squared_error(x_valid_unscaled, results))
                metrics_df = pd.DataFrame({'Metric': ['MAE', 'RMSE'], 'Value': [mae, rmse]})
                st.session_state['metrics_df'] = metrics_df  # Store the DataFrame in session state
        else:
            st.warning('Please train the model first before evaluating')

    # Main Panel
    # Check if metrics are available and display them
    if 'metrics_df' in st.session_state:
        with st.expander("Model Evaluation Metrics"):
            st.table(st.session_state['metrics_df'].T)  # Transpose the DataFrame to make it a one-line table

    # Plot and Generate Predictions Button
    if st.button('Plot and Generate Predictions'):
        if 'cnn_rnn_model' in st.session_state:
            st.session_state['generate_predictions'] = True
            st.session_state['show_spinner'] = True
        else:
            st.warning('Please train the model first before generating predictions')

if 'show_spinner' in st.session_state and st.session_state['show_spinner']:
    with st.spinner('Generating Predictions...'):
        st.session_state['show_spinner'] = False

if 'generate_predictions' in st.session_state and st.session_state['generate_predictions']:
    # Loading model and related data from session state
    cnn_rnn_model = st.session_state['cnn_rnn_model']
    scaler = st.session_state['scaler']
    scaled_series = st.session_state['scaled_series']
    window_size = st.session_state['window_size']

    # Generating future predictions
    future_years = np.arange(2021, 2031, 1)
    scaled_series_future = scaled_series[-window_size:]
    future_forecast = []
    for time in range(10):
        prediction = cnn_rnn_model.predict(scaled_series_future[np.newaxis, :, np.newaxis])
        future_forecast.append(prediction[0, 0])
        scaled_series_future = np.append(scaled_series_future, prediction)
        scaled_series_future = scaled_series_future[-window_size:]

    # Inversing the scaling transformation and rounding the results
    forecasted_values = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten().round().astype(
        int)

    # Create Plotly traces
    historical_trace = go.Scatter(x=df['YEAR'], y=df[selected_category], mode='lines+markers',
                                  name='Historical Data')
    prediction_trace = go.Scatter(x=future_years, y=forecasted_values, mode='lines+markers', name='Predictions')

    # Create Plotly layout with adjusted title settings and additional layout configurations
    layout = go.Layout(
        title={
            'text': f'Historical Data and Predictions for {country} - {selected_category}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis=dict(
            title='',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title='tonnes',
            titlefont=dict(size=18, family="Arial, sans-serif", color="#555555")
        ),
        template="plotly_dark",
        font=dict(family="Arial, sans-serif", size=18, color="#555555"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,  # Adjust height as needed
        showlegend=True
    )

    # Create Plotly figure
    fig = go.Figure(data=[historical_trace, prediction_trace], layout=layout)

    # Display Plotly figure in the main panel
    st.plotly_chart(fig, use_container_width=True)

    # Display Predictions in a table
    future_years = np.arange(2021, 2031, 1).astype(int)  # Ensure future_years is an array of integers
    predictions_df = pd.DataFrame({'Year': future_years, 'Predicted Values': forecasted_values})
    predictions_df['Predicted Values'] = predictions_df['Predicted Values'].round().astype(
        int)  # Round predictions to nearest integer

    # Use Styler to set table styles
    styled_df = predictions_df.style.format({"Year": "{:0>4}", "Predicted Values": "{:0>4}"}) \
        .set_properties(**{
        'text-align': 'center',
        'font-size': '10pt',  # Adjusted font size
        'border-style': 'solid',
        'border-width': '1px',
        'width': '50px',  # Adjusted width of each cell
        'height': '20px',  # Adjusted height of each cell
    }) \
        .set_table_styles([{
        'selector': 'th',
        'props': [('font-size', '12pt'), ('text-align', 'center')]
    }])

    # Use st.write() with unsafe_allow_html to allow custom HTML and CSS
    st.write(styled_df.to_html(index=False), unsafe_allow_html=True)

    # Reset the generate_predictions flag
    st.session_state['generate_predictions'] = False
