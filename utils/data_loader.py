import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# Function is NOT executed in 'main.py' script. 
# Downloads the images from scracth and applies ResNe50 for feature extraction.
# Make sure to have at least 60 GB of free disk space to load the COCO captions dataset; it will be saved as '.tfrecord' data.
# Otherwise, the exracted features are already saved for training under "data/processed_data" directory. 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_data')
SHUFFLE_BUFFER_SIZE = 10000

if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

def serialize_example(feature):
    feature = {
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=feature))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def save_preprocessed_data(dataset, name):
    path = os.path.join(PROCESSED_DATA_DIR, f"{name}.tfrecord")
    with tf.io.TFRecordWriter(path) as writer:
        for data in dataset:
            serialized_example = serialize_example(data.numpy())
            writer.write(serialized_example)

def load_and_preprocess_data():
    data_directory = os.path.join(BASE_DIR, "data")
    os.makedirs(data_directory, exist_ok=True)

    train_data, test_data, val_data = tfds.load(
        "coco_captions", data_dir=data_directory, split=["train", "test", "val"]
    )

    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    resnet.trainable = False

    def preprocess_data(element):
        image = element["image"]
        image = tf.image.resize(image, (224, 224))
        image = preprocess_input(image)
        image = tf.expand_dims(image, 0)
        image = resnet(image, training=False)
        image = tf.squeeze(image, axis=0)
        return image

    train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.shuffle(SHUFFLE_BUFFER_SIZE).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_data = val_data.shuffle(SHUFFLE_BUFFER_SIZE).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    return train_data, test_data, val_data

train_data, test_data, val_data = load_and_preprocess_data()

save_preprocessed_data(train_data, "train_data")
save_preprocessed_data(test_data, "test_data")
save_preprocessed_data(val_data, "val_data")