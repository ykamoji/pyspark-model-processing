import sys
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from utils import get_label_map
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import time

dataset_path = '/Users/ykamoji/Documents/ImageDatabase/cifar-10-batches-py/'
label_map = get_label_map(dataset_path)

processor = ViTImageProcessor.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',
                                                  cache_dir='models/')
model = ViTForImageClassification.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',
                                                      cache_dir='models/')
model.eval()
device = torch.device('cpu')
model.to(device)

def reshape_image(record):
        image, label = record
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


def calc_acc(record):
        return record[0], f"{float(sum(record[1]) * 100 / len(record[1])):.2f}%"

def processBatch(batch_df, batch_id):
        ## The same stuff from main.py can be done here now.
        results = batch_df.sparkSession.sparkContext.parallelize(batch_df.rdd.collect())\
                .map(reshape_image).map(predictImage).collect()

        print("-"*50)
        print(f"Batch {batch_id}")
        print("-" * 50)
        for class_label, pred in results:
                print(f"{class_label} : {pred}")
        print("-" * 50)
        print("-" * 50+"\n\n")


def start_streaming():
        spark = SparkSession.builder. \
                appName("ImageDataStream"). \
                master("local[*]"). \
                config("spark.executor.memory", "16G"). \
                config("spark.driver.memory", "16G"). \
                config("spark.sql.streaming.forceDeleteTempCheckpointLocation", True).\
                getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

        # spark.conf.set("spark.sql.streaming.schemaInference", True)

        imaga_data_schema = StructType([
                StructField("data", ArrayType(elementType=IntegerType()), False),
                StructField("label", IntegerType(), False),
        ])


        streaming_df = spark.readStream.format("json"). \
                schema(imaga_data_schema).\
                option("cleanSource", "archive").\
                option("sourceArchiveDir", "streams/archive").\
                option("maxFilesPerTrigger", 1).\
                load("streams/input")

        streaming_df.printSchema()

        query = streaming_df

        query.writeStream.foreachBatch(processBatch).start()

        time.sleep(120)
        query.stop()
        spark.stop()

if __name__ == '__main__':
        start_streaming()


