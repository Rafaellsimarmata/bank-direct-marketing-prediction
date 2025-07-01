import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from keras_tuner.engine import base_tuner
from transform import (CATEGORICAL_FEATURE_KEYS, INT_FEATURE_KEYS, FLOAT_FEATURE_KEYS, LABEL_KEY, transformed_name)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from typing import Any, Dict, NamedTuple, Text

NUM_EPOCHS = 10

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

def get_class_weights(dataset, batch_size=64):
    """Calculate class weights from TF Dataset"""
    y_train = []
    for features, labels in dataset.unbatch().batch(batch_size):
        y_train.extend(labels.numpy())
    y_train = np.array(y_train).flatten()
    # print(y_train)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return {0: class_weights[0], 1: class_weights[1]}

def get_early_stopping(monitor='val_auc', mode='max'):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        verbose=1,
        patience=10,
    )

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
 
def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """Get post_transform feature & create batches of data"""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

def get_model_tuner(hyperparameters):
    """Build model with hyperparameter tuning and imbalance handling"""
    # Model architecture parameters
    num_hidden_layers = hyperparameters.Choice("num_hidden_layers", values=[1, 2, 3])
    dense_unit = hyperparameters.Int("dense_unit", min_value=16, max_value=256, step=32)
    dropout_rate = hyperparameters.Float("dropout_rate", min_value=0.1, max_value=0.9, step=0.1)
    learning_rate = hyperparameters.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
    # Imbalance handling parameters
    use_class_weight = hyperparameters.Boolean("use_class_weight")
    weight_ratio = hyperparameters.Float("weight_ratio", min_value=1.0, max_value=10.0, step=0.5) if use_class_weight else 1.0

    # Input layers
    input_features = []
    for key, dim in CATEGORICAL_FEATURE_KEYS.items():
        input_features.append(layers.Input(shape=(dim+1,), name=transformed_name(key)))
    for feature in INT_FEATURE_KEYS:
        input_features.append(layers.Input(shape=(1,), name=transformed_name(feature)))
    for feature in FLOAT_FEATURE_KEYS:
        input_features.append(layers.Input(shape=(1,), name=transformed_name(feature)))

    # Model architecture
    concatenate = layers.concatenate(input_features)
    deep = layers.Dense(dense_unit, activation=tf.nn.relu)(concatenate)

    for _ in range(num_hidden_layers):
        deep = layers.Dense(dense_unit, activation=tf.nn.relu)(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(deep)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    if use_class_weight:
        def weighted_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            weight_vector = y_true * weight_ratio + (1 - y_true) * 1.0
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            weighted_bce = weight_vector * bce
            return tf.reduce_mean(weighted_bce)
        loss_fn = weighted_loss
    else:
        loss_fn = 'binary_crossentropy'

    model.compile(
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def tuner_fn(fn_args):
    """Build the tuner with imbalance handling"""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Load datasets
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
    eval_set = input_fn(fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS)

    # Calculate class weights
    class_weight_dict = get_class_weights(train_set)

    tuner = kt.Hyperband(
        hypermodel=get_model_tuner,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=NUM_EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kt_hyperband_imbalanced",
        overwrite=True
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "class_weight": class_weight_dict, 
            "callbacks": [
                get_early_stopping(monitor='val_auc', mode='max'),
                # tf.keras.callbacks.TensorBoard(log_dir=fn_args.working_dir)
            ]
        },
    )