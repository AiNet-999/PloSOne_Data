from contextlib import closing
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Load the data
closing_prices = pd.read_csv('/kaggle/input/tetsingdata/SP500_Closing_Prices.csv', header=None).fillna(method='ffill').fillna(method='bfill')

sentiment_scores1 = pd.read_csv('/kaggle/input/mynewdataset/Sentiment/sentiment_repeated_gaza.csv', header=None).fillna(method='ffill').fillna(method='bfill')


adjacency_matrix_short = pd.read_csv('/kaggle/input/testing/short_term_adjacency.csv', header=None).to_numpy()
adjacency_matrix_medium = pd.read_csv('/kaggle/input/testing/medium_term_adjacency.csv', header=None).to_numpy()
adjacency_matrix_long = pd.read_csv('/kaggle/input/testing/long_term_adjacency.csv', header=None).to_numpy()

num_repeats = 501

def repeat_first_column(df, num_repeats):
    df_repeated = df[[0]].copy()  
    df_repeated = pd.concat([df_repeated] * num_repeats, axis=1)
    return df_repeated
zeros = pd.DataFrame(np.zeros((90, 1)), columns=[0])
sentiment_scores1 = pd.concat([sentiment_scores1, zeros], ignore_index=True)
sentiment_scores1= repeat_first_column(sentiment_scores1, num_repeats)


closing_prices = closing_prices.to_numpy()
sentiment_scores1 = sentiment_scores1.to_numpy()


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices = scaler.fit_transform(closing_prices)

print("Final Shapes---->")
print(closing_prices.shape)

print(sentiment_scores1.shape)

# Stack the closing prices and sentiment scores to create a multi-feature input
data_array = np.stack([closing_prices,sentiment_scores1], axis=-1)

def preprocess(data_array, train_size, val_size):
    num_time_steps = data_array.shape[0]
    num_train, num_val = int(num_time_steps * train_size), int(num_time_steps * val_size)

    train_array = data_array[:num_train]
    val_array = data_array[num_train : (num_train + num_val)]
    test_array = data_array[(num_train + num_val) :]

    return train_array, val_array, test_array

train_array, val_array, test_array = preprocess(data_array, train_size=0.7, val_size=0.1)

def create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle=True, multi_horizon=False):
    inputs = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[:-forecast_horizon], None, sequence_length=input_sequence_length, shuffle=False, batch_size=batch_size
    )

    target_offset = input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[target_offset:, :, 0], None, sequence_length=target_seq_length, shuffle=False, batch_size=batch_size
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    print(f"Input Shape: {data_array[:-forecast_horizon].shape}")
    print(f"Target Shape: {data_array[target_offset:, :, 0].shape}")
    return dataset.prefetch(16).cache()

in_feat = 2
out_feat = 10
lstm_units = 64
input_sequence_length = 1
forecast_horizon = 1
batch_size = 64

train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size)
val_dataset = create_tf_dataset(val_array, input_sequence_length, forecast_horizon, batch_size)
test_dataset = create_tf_dataset(test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False)


        

# Graph convolutional network layers
class GraphConv(layers.Layer):
    def __init__(self, in_feat, out_feat, graph_info, aggregation_type="mean", combination_type="concat", activation=None, **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        
        # Define a learnable adjacency matrix
        self.adjacency_matrix = self.add_weight(
            shape=(graph_info.num_nodes, graph_info.num_nodes),  # square matrix
            initializer=keras.initializers.GlorotUniform(),  # or another initializer
            trainable=True,
            name="learnable_adjacency_matrix"
        )

        # Apply sigmoid activation to keep the adjacency matrix values between 0 and 1
        self.sigmoid = layers.Activation('sigmoid')
        
        # Weight matrix for the GraphConv
        self.weight = tf.Variable(initial_value=keras.initializers.glorot_uniform()(shape=(in_feat, out_feat), dtype="float32"), trainable=True)
        self.activation = layers.Activation(activation) if activation else None

    def aggregate(self, neighbour_representations):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)
        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )
        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features):
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features):
        # Use the learnable adjacency matrix (with sigmoid activation) for graph convolution
        adjacency_matrix = self.sigmoid(self.adjacency_matrix)
        
        # Get the neighbor representations using the adjacency matrix
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation, aggregated_messages):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")
        return self.activation(h) if self.activation else h

    def call(self, features):
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)
# LSTM with graph convolution
class LSTMGC(layers.Layer):
    def __init__(self, in_feat, out_feat, lstm_units, input_seq_len, output_seq_len, graph_info, graph_conv_params=None, **kwargs):
        super().__init__(**kwargs)
        if graph_conv_params is None:
            graph_conv_params = {"aggregation_type": "mean", "combination_type": "concat", "activation": None}
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        # Stacked LSTM: 2 layers
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, activation="relu")
        self.lstm2 = layers.LSTM(32, activation="relu", dropout=0.2)

        self.dense = layers.Dense(output_seq_len)
        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len


    def call(self, inputs):
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(inputs)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = shape[0], shape[1], shape[2], shape[3]
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))

        # Stacked LSTM
        x = self.lstm1(gcn_out)
        x = self.lstm2(x)

        dense_output = self.dense(x)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(output, [1, 2, 0])


# GRU with graph convolution
class GRUGC(layers.Layer):
    def __init__(self, in_feat, out_feat, gru_units, input_seq_len, output_seq_len, graph_info, graph_conv_params=None, **kwargs):
        super().__init__(**kwargs)
        if graph_conv_params is None:
            graph_conv_params = {"aggregation_type": "mean", "combination_type": "concat", "activation": None}
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        # Stacked GRU: 2 layers
        self.gru1 = layers.GRU(gru_units, return_sequences=True, activation="relu")
        self.gru2 = layers.GRU(32, activation="relu", dropout=0.2)

        self.dense = layers.Dense(output_seq_len)
        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    
    def call(self, inputs):
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(inputs)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = shape[0], shape[1], shape[2], shape[3]
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))

        # Stacked GRU
        x = self.gru1(gcn_out)
        x = self.gru2(x)

        dense_output = self.dense(x)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(output, [1, 2, 0])


# Bidirectional LSTM with graph convolution
class BiLSTMGC(layers.Layer):
    def __init__(self, in_feat, out_feat, lstm_units, input_seq_len, output_seq_len, graph_info, graph_conv_params=None, **kwargs):
        super().__init__(**kwargs)
        if graph_conv_params is None:
            graph_conv_params = {"aggregation_type": "mean", "combination_type": "concat", "activation": None}
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        # Stacked Bidirectional LSTM: 2 layers
        self.bilstm1 = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, activation="relu"))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(32, activation="relu", dropout=0.2))

        self.dense = layers.Dense(output_seq_len)
        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len



    def call(self, inputs):
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(inputs)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = shape[0], shape[1], shape[2], shape[3]
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))

        # Stacked BiLSTM
        x = self.bilstm1(gcn_out)
        x = self.bilstm2(x)

        dense_output = self.dense(x)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(output, [1, 2, 0])








class GraphInfo:
    def __init__(self, edges, num_nodes):
        self.edges = edges
        self.num_nodes = num_nodes

def compute_adjacency_matrix(route_distances, epsilon):
    adjacency_matrix = np.where(route_distances >= epsilon, 1, 0)
    return adjacency_matrix

print("Short_term Adjacency Matrix")
adjacency_matrix_short= compute_adjacency_matrix(adjacency_matrix_short, epsilon=0.5)
node_indices, neighbor_indices = np.where(adjacency_matrix_short == 1)
graph1 = GraphInfo(edges=(node_indices.tolist(), neighbor_indices.tolist()), num_nodes=adjacency_matrix_short.shape[0])
num_edges = len(node_indices) 
print(f"Total number of edges in the graph: {num_edges}")

print("Medium_term Adjacency Matrix")

adjacency_matrix_medium = compute_adjacency_matrix(adjacency_matrix_medium, epsilon=0.5)
node_indices, neighbor_indices = np.where(adjacency_matrix_medium == 1)
graph2 = GraphInfo(edges=(node_indices.tolist(), neighbor_indices.tolist()), num_nodes=adjacency_matrix_medium.shape[0])
num_edges = len(node_indices) 
print(f"Total number of edges in the graph: {num_edges}")

print("Long_term Adjacency Matrix")
adjacency_matrix_long = compute_adjacency_matrix(adjacency_matrix_long, epsilon=0.5)
node_indices, neighbor_indices = np.where(adjacency_matrix_long == 1)
graph3 = GraphInfo(edges=(node_indices.tolist(), neighbor_indices.tolist()), num_nodes=adjacency_matrix_long.shape[0])
num_edges = len(node_indices) 
print(f"Total number of edges in the graph: {num_edges}")

st_gcn = LSTMGC(
    in_feat=in_feat,
    out_feat=out_feat,
    lstm_units=lstm_units,
    input_seq_len=input_sequence_length,
    output_seq_len=forecast_horizon,
    graph_info=graph1
  
  #  graph_conv_params={"aggregation_type": best_aggregation_type, "combination_type": best_combination_type}
)
gcn_gru_model = GRUGC(
    in_feat=in_feat,
    out_feat=out_feat,
    gru_units=lstm_units, 
    input_seq_len=input_sequence_length,
    output_seq_len=forecast_horizon,
    graph_info=graph3

)

bilstm_model = BiLSTMGC(
    in_feat=in_feat,
    out_feat=out_feat,
    lstm_units=lstm_units,  
    input_seq_len=input_sequence_length,
    output_seq_len=forecast_horizon,
    graph_info=graph2

)
inputs = layers.Input((int(input_sequence_length), graph1.num_nodes, int(in_feat)))
lstm_outputs = st_gcn(inputs)
gru_outputs = gcn_gru_model(inputs)
bilstm_outputs=bilstm_model (inputs)

# Average the outputs from all models
average_outputs = layers.Average()([lstm_outputs, gru_outputs,bilstm_outputs])


model = keras.models.Model(inputs, average_outputs )
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')


# Train the final model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=200)


# Print final training history
print("\nFinal Training History:")
print(f"Final Training Loss: {history.history['loss'][-1]}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]}")
print("Fitness History (each epoch):")


import numpy as np
import matplotlib.pyplot as plt
stock_symbols = [
    'AAL', 'AAP', 'AAPL', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 
    'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 
    'ALGN', 'ALK', 'ALL', 'AMAT','AMD','AME', 'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANSS', 'AON', 'AOS'

]

stock_symbols = [
    'VRSN', 'VRTX', 'VTR', 'VZ', 'WAB', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL',
    'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WU', 'WY', 'WYNN', 'XEL', 'XOM',
    'XRAY', 'XRX', 'YUM', 'ZBH', 'ZION'
]


x_test, y = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)

plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, 0])
plt.plot(y_pred[:, 0, 0])
plt.legend(["actual", "forecast"])

# Compute Mean Absolute Error (MAE)

naive_mae = np.abs(x_test[:, -1, :, 0] - y[:, 0, :]).mean()
model_mae = np.abs(y_pred[:, 0, :] - y[:, 0, :]).mean()

# Compute Mean Squared Error (MSE)
naive_mse = np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean()
model_mse = np.square(y_pred[:, 0, :] - y[:, 0, :]).mean()

# Compute Root Mean Squared Error (RMSE)
naive_rmse = np.sqrt(naive_mse)
model_rmse = np.sqrt(model_mse)

# Round values to three decimal places
naive_mae = round(naive_mae, 5)
model_mae = round(model_mae, 5)
naive_mse = round(naive_mse, 5)
model_mse = round(model_mse, 5)
naive_rmse = round(naive_rmse, 5)
model_rmse = round(model_rmse, 5)


print(f"Naive MAE: {naive_mae}, Model MAE: {model_mae}")
print(f"Naive MSE: {naive_mse}, Model MSE: {model_mse}")
print(f"Naive RMSE: {naive_rmse}, Model RMSE: {model_rmse}")
