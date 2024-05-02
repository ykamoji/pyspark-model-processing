import pickle
from pyspark.sql.types import StructType, IntegerType, StructField, TimestampType, FloatType, StringType
from pyspark.sql.functions import col

conv_model_list = [
    "facebook/convnext-tiny-224",
    "facebook/convnext-small-224",
    "facebook/convnext-base-224",
    "facebook/convnext-large-224",
    "facebook/convnextv2-nano-22k-224",
    "facebook/convnextv2-tiny-22k-224",
    "facebook/convnextv2-base-22k-224",
    "facebook/convnextv2-large-22k-224",
    "facebook/convnextv2-huge-1k-224"
]

vit_model_list = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/deit-tiny-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/deit-base-patch16-224",
    "facebook/deit-tiny-distilled-patch16-224",
    "facebook/deit-base-distilled-patch16-224"
]

streaming_schema = StructType([
        StructField("batch_id",IntegerType(), False),
        StructField("batch_size",IntegerType(), False),
        StructField("triggered",TimestampType(), False),
        StructField("start", TimestampType()),
        StructField("end", TimestampType()),
        StructField("accuracy", FloatType(), False),
        StructField("env", StringType(), False),
])

model_details = {
    "facebook/convnext-tiny-224": {"params": 28.589, "size": 109.059},
    "facebook/convnext-small-224": {"params": 50.224, "size": 191.588},
    "facebook/convnext-base-224": {"params": 88.591, "size": 337.950},
    "facebook/convnext-large-224": {"params": 197.767, "size": 754.423},
    "facebook/convnextv2-nano-22k-224": {"params": 15.624, "size": 59.600},
    "facebook/convnextv2-tiny-22k-224": {"params": 28.635, "size": 109.236},
    "facebook/convnextv2-base-22k-224": {"params": 88.718, "size": 338.432},
    "facebook/convnextv2-large-22k-224": {"params": 197.957, "size": 755.145},
    "facebook/convnextv2-huge-1k-224": {"params": 660.290, "size": 2518.805},
    "google/vit-base-patch16-224": {"params": 86.568, "size": 330.229},
    "google/vit-large-patch16-224": {"params": 304.327, "size": 1160.914},
    "facebook/deit-tiny-patch16-224": {"params": 5.717, "size": 21.810},
    "facebook/deit-small-patch16-224": {"params": 22.051, "size": 84.117},
    "facebook/deit-base-patch16-224": {"params": 86.568, "size": 330.229},
    "facebook/deit-tiny-distilled-patch16-224": {"params": 5.911, "size": 22.548},
    "facebook/deit-base-distilled-patch16-224": {"params": 87.338, "size": 333.169},
}


def createDataSet(dataset_path):
    train_dataset = []
    test_dataset = []
    for i in range(1, 6):
        data = pickle.load(open(dataset_path + f'data_batch_{i}', 'rb'), encoding='latin-1')
        train_dataset.extend(zip(data["data"], data["labels"]))

    test_data = pickle.load(open(dataset_path + f'test_batch', 'rb'), encoding='latin-1')
    test_dataset.extend(zip(test_data["data"], test_data["labels"]))

    label_map = get_label_map(dataset_path)

    return train_dataset, test_dataset, label_map


def get_label_map(dataset_path):
    meta = pickle.load(open(dataset_path + f'batches.meta', 'rb'), encoding='latin-1')
    return {index: label for index, label in enumerate(meta['label_names'])}


def collate_fn(images, labels):
    return {
        'pixel_values': images['pixel_values'],
        'labels': labels
    }


def select_cast(column):
    return col(column).cast(TimestampType())