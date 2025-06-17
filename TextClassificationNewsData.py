# Import necessary libraries
import numpy as np                         # For numerical operations
import tensorflow as tf                    # TensorFlow library for building and training ML models
import tensorflow_hub as hub               # For loading pre-trained models from TensorFlow Hub
import tensorflow_datasets as tfds         # For loading built-in datasets
import tf_keras                            # Alias for Keras inside TensorFlow for building deep learning models

# Load AG News dataset and split into training, validation, and test sets
train_data, validation_data, test_data = tfds.load(
    name="ag_news_subset",                         # Dataset name
    split=('train[:60%]', 'train[60%:]', 'test'),  # Splitting into 60% train, 40% validation, and test set
    as_supervised=True                             # Returns (text, label) pairs
)

# Fetch a batch of training examples for inspection
train_example_batch, train_lables_batch = next(iter(train_data.batch(10)))

# Define a text embedding layer using TensorFlow Hub (Swivel embedding model)
hub_layer = hub.KerasLayer(
    "https://kaggle.com/models/google/gnews-swivel/frameworks/TensorFlow2/variations/tf2-preview-20dim/versions/1",
    output_shape=[20],      # Embedding vector size
    input_shape=[],         # Input is a scalar string
    dtype=tf.string,        # Data type is string
    trainable=True          # Allow fine-tuning during training
)

# Test the embedding layer on a batch of examples
hub_layer(train_example_batch[0:])

# Redefine the embedding layer from a different URL (alternate model)
hub_layer = hub.KerasLayer(
    "https://www.kaggle.com/models/google/gnews-swivel/TensorFlow2/tf2-preview-20dim/1", 
    output_shape=[20],
    input_shape=[], 
    dtype=tf.string
)

# Build the Sequential model
model = tf_keras.Sequential()
model.add(hub_layer)                             # Add pre-trained text embedding layer
model.add(tf_keras.layers.Dense(16, activation='relu'))  # Hidden dense layer with ReLU activation
model.add(tf_keras.layers.Dense(1, activation='sigmoid')) # Output layer for binary classification

# Print model architecture summary
model.summary()

# Compile the model
model.compile(
    optimizer='adam',                                    # Optimizer
    loss=tf_keras.losses.BinaryCrossentropy(from_logits=True),  # Binary cross-entropy loss
    metrics=['accuracy']                                 # Track accuracy
)

# Train the model
history = model.fit(
    train_data.shuffle(10000).batch(100),                # Shuffle and batch the training data
    epochs=25,                                            # Train for 25 epochs
    validation_data=validation_data.batch(100),          # Use validation data
    verbose=1                                             # Print training progress
)

# Evaluate model performance on the test set
results = model.evaluate(test_data.batch(100), verbose=2)

# Print each metric result
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
