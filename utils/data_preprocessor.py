import os
import tensorflow as tf
from configs.settings import SUBSET_SIZE


# Define directory constants.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_data')

# Description of the dataset features.
FEATURE_DESCRIPTION = {
    'data': tf.io.FixedLenFeature([2048], tf.float32) 
}

def _parse_function(example_proto):
    """Parses the input `tf.Example` proto into the dictionary of tensors."""
    parsed_example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    return parsed_example['data']

def load_preprocessed_data(name):
    """Load the preprocessed data from a tfrecord file."""
    path = os.path.join(PROCESSED_DATA_DIR, f"{name}.tfrecord")
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function)
    return dataset

def create_subsets(data):
    """
    Create two random subsets (R1 and R2) of the data with overlapping and non-overlapping portions (target vectors t1 and t2).
    """
    tf.debugging.assert_greater_equal(len(data), 2*SUBSET_SIZE, 
                                      message="Input BATCH_SIZE should be at least 2*SUBSET_SIZE"
    )

    data = tf.random.shuffle(data)

    # Determine sizes of shared and unique parts of subsets.
    R_shared_size = tf.random.uniform(shape=(), minval=0, maxval=SUBSET_SIZE, dtype=tf.int32)
    R_fill_size = SUBSET_SIZE - R_shared_size

    # Split the data into overlapping and non-overlapping parts.
    R_shared = data[:R_shared_size]
    R_fill = data[R_shared_size:]
    R1_fill = R_fill[:R_fill_size]
    R2_fill = R_fill[R_fill_size: 2*R_fill_size]
    
    # Create subsets.
    R1 = tf.concat([R_shared, R1_fill], axis=0)
    R2 = tf.concat([R_shared, R2_fill], axis=0)
    t1 = tf.concat([tf.ones(R_shared_size, dtype=tf.float32), tf.zeros(R_fill_size, dtype=tf.float32)], axis=0)
    t2 = tf.concat([tf.ones(R_shared_size, dtype=tf.float32), tf.zeros(R_fill_size, dtype=tf.float32)], axis=0)
    
    # Shuffle the subsets.
    perm1 = tf.random.shuffle(tf.range(SUBSET_SIZE))
    R1, t1 = tf.gather(R1, perm1), tf.gather(t1, perm1)
    perm2 = tf.random.shuffle(tf.range(SUBSET_SIZE))
    R2, t2 = tf.gather(R2, perm2), tf.gather(t2, perm2)

    return R1, t1, R2, t2

def prepare_data(BATCH_SIZE, train="train_data", test="test_data", val="val_data"):
    """
    Prepare the data for training, testing, and validation.
    Load the data, batch it, and apply the subset creation.
    """
    train_data = load_preprocessed_data(train)
    test_data = load_preprocessed_data(test)
    val_data = load_preprocessed_data(val)

    train_dataset = train_data.batch(2*SUBSET_SIZE, drop_remainder=True).map(create_subsets)
    test_dataset = test_data.batch(2*SUBSET_SIZE, drop_remainder=True).map(create_subsets)
    val_dataset = val_data.batch(2*SUBSET_SIZE, drop_remainder=True).map(create_subsets)

    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset, val_dataset

