import tensorflow as tf
import tensorflow_transform as tft 
from keras.layers import Dense, Concatenate, Dropout
import os  
import tensorflow_hub as hub
from keras.utils.vis_utils import plot_model
from tfx.components.trainer.fn_args_utils import FnArgs
 
LABEL_KEY = "y"

CATEGORICAL_FEATURE_KEYS = {
    'job': 12,
    'marital': 4,
    'education': 8,
    'default': 3,
    'housing': 3,
    'loan': 3,
    'contact': 2,
    'month': 10,
    'day_of_week': 5,
    'poutcome': 3
}

NUMERICAL_FEATURE_KEYS =['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
 
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
 
 
def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    # Get post_transform feature 10000 
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset
 
def model_builder(hyperparameters):
    list_feature = []

    for key, dim in CATEGORICAL_FEATURE_KEYS.items():
        inputs = tf.keras.layers.Input(shape = (dim+1, ), name = transformed_name(key))
        list_feature.append(inputs)
    
    for key in NUMERICAL_FEATURE_KEYS:
        inputs =  tf.keras.layers.Input(shape=(1,), name=transformed_name(key))
        list_feature.append(inputs)

    concatenate = Concatenate()(list_feature)
    
    x = Dense(hyperparameters["dense_unit"], activation="relu")(concatenate)
    
    for _ in range(hyperparameters["num_hidden_layers"]):
        x = Dense(
            hyperparameters["dense_unit"], activation="relu")(x)
        x = Dropout(hyperparameters["dropout_rate"])(x)
    
    outputs = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=list_feature, outputs = outputs)
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model 
 
def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        # get predictions using the transformed features
        return model(transformed_features)
        
    return serve_tf_examples_fn
    
def run_fn(fn_args: FnArgs) -> None:
    
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    # Build the model
    hyperparameters = fn_args.hyperparameters["values"]
    model = model_builder(hyperparameters)
    
    # Train the model
    model.fit(x = train_set,
            validation_data = val_set,
            callbacks = [tensorboard_callback, es, mc],
            # steps_per_epoch = 100, 
            # validation_steps= 100,
            epochs=10)
    
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
    
    plot_model(
        model, 
        to_file='images/model_plot.png', 
        show_shapes=True, 
        show_layer_names=True
    )