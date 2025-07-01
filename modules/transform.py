import tensorflow as tf
import tensorflow_transform as tft

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

INT_FEATURE_KEYS = ['age', 'campaign', 'pdays', 'previous']
FLOAT_FEATURE_KEYS = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def one_hot_encoding(indices, num_label):
    one_hot = tf.one_hot(indices, num_label) 
    return tf.reshape(one_hot, (-1, num_label)) 

# Special handling for pdays (999 means no previous contact)
def _fill_missing_pdays(pdays):
    return tf.where(tf.equal(pdays, 999), tf.cast(-1, tf.int64), pdays)

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Returns:
        outputs: map from feature keys to transformed features.    
    """
    outputs = {}
        
    # Process categorical features
    for key, value in CATEGORICAL_FEATURE_KEYS.items():
        integerized = tft.compute_and_apply_vocabulary(inputs[key], top_k = value+1)
        outputs[transformed_name(key)] = one_hot_encoding(integerized, value + 1)

    # Standardize numerical features
    for key in INT_FEATURE_KEYS:
        if key == 'pdays':
            # Special handling for pdays
            filled = _fill_missing_pdays(inputs[key])
            scaled = tft.scale_to_z_score(filled)
            outputs[transformed_name(key)] = tf.where(
                tf.equal(filled, -1), tf.cast(-2.0, tf.float32), scaled)
        else:
            value = tf.cast(inputs[key], tf.int64)
            outputs[transformed_name(key)] = tft.scale_to_z_score(value)

    for key in FLOAT_FEATURE_KEYS:
        value = tf.cast(inputs[key], tf.float32)
        outputs[transformed_name(key)] = tft.scale_to_z_score(value)

    # Transform the 'y' (label) column
    label = inputs[LABEL_KEY]
    label_int = tf.where(tf.equal(label, 'yes'), 1, 0)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(label_int, tf.int64)
    
    return outputs
