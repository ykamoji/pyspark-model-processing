import os.path
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import numpy as np
from utils import get_label_map, select_cast
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
import time

dataset_path = '/Users/ykamoji/Documents/ImageDatabase/cifar-10-batches-py/'
tracker_path = 'logs/tracker.json'

label_map = get_label_map(dataset_path)

processor = ViTImageProcessor.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',
                                              cache_dir='models/')
model = ViTForImageClassification.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',
                                                  cache_dir='models/')
model.eval()
device = torch.device('cpu')
model.to(device)


def reshape_image(record):
    batch_id, image, label = record
    height = 32
    width = 32
    nChannels = 3
    data = [float(x) for x in np.array(image).reshape(nChannels, height, width).transpose(1, 2, 0).flatten()]
    return data, label


def predictImage(record):
    gt = record[1]
    class_label = label_map[gt]
    image = np.array(record[0]).reshape(32, 32, 3).astype(np.uint8)
    input = processor(images=image, return_tensors="pt")
    input = input.to(device)
    output = model(**input)
    logits = output.logits.cpu()
    pred = logits.argmax(-1).item()
    return class_label, True if gt == pred else False

def processBatch(batch_df, query_batch_id):

    batch_id = batch_df.select("idx").first().idx

    if not batch_id:
        batch_id = query_batch_id

    start = time.time()

    results = batch_df.sparkSession.sparkContext.parallelize(batch_df.rdd.collect()) \
        .map(reshape_image).map(predictImage).collect()

    end = time.time()

    print("-" * 50)
    print(f"Batch {batch_id}")
    print("-" * 50)
    accuracy = 0
    for class_label, pred in results:
        print(f"{class_label} : {pred}")
        if pred:
            accuracy += 1

    accuracy /= len(results)
    accuracy = round(accuracy*100, 3)
    print("-" * 50)
    print("-" * 50 + "\n\n")

    tracker_df = batch_df.sparkSession.read.json(tracker_path)

    tracker_df = tracker_df.withColumn("start",
                                       F.when(tracker_df.batch_id == batch_id, start).otherwise(tracker_df.start))

    tracker_df = tracker_df.withColumn("end",
                                       F.when(tracker_df.batch_id == batch_id, end).otherwise(tracker_df.end))

    tracker_df = tracker_df.withColumn("accuracy",
                                       F.when(tracker_df.batch_id == batch_id, accuracy).otherwise(tracker_df.accuracy))

    tracker_df.select(F.col("batch_id"), F.col("batch_size"), F.col("accuracy"), select_cast('triggered'),
                      select_cast('start'), select_cast('end'), F.col("env")).\
                withColumn("start", F.when(F.col("start").cast(IntegerType()) == 0, "-").\
                           otherwise((F.col("start")))). \
                withColumn("end", F.when(F.col("end").cast(IntegerType()) == 0, "-").\
                            otherwise((F.col("end")))).\
                show(truncate=False)

    tracker_df.toPandas().to_json(tracker_path, orient='records', force_ascii=False, lines=True)

def start_streaming():
    spark = SparkSession.builder. \
        appName("ImageDataStream"). \
        master("local[*]"). \
        config("spark.ui.port", "4050"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True). \
        getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # spark.conf.set("spark.sql.streaming.schemaInference", True)

    imaga_data_schema = StructType([
        StructField("idx", IntegerType(), False),
        StructField("data", ArrayType(elementType=IntegerType()), False),
        StructField("label", IntegerType(), False),
    ])

    streaming_df = spark.readStream.format("json"). \
        schema(imaga_data_schema). \
        option("cleanSource", "archive"). \
        option("sourceArchiveDir", "streams/archive"). \
        option("maxFilesPerTrigger", 1). \
        load("streams/input")

    streaming_df.printSchema()

    keep_checking = True
    while keep_checking:
        try:
            if os.path.exists(tracker_path):
                keep_checking = False
                time.sleep(2)
        except:
            keep_checking = True

    query = streaming_df

    query.writeStream.foreachBatch(processBatch).start()

    time.sleep(600)
    spark.stop()


if __name__ == '__main__':
    start_streaming()
