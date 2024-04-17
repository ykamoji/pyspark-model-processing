import glob
import os
import shutil
import random
import time
import json
from pyspark.sql import SparkSession
from utils import createDataSet, schema

dataset_path = '/Users/ykamoji/Documents/ImageDatabase/cifar-10-batches-py/'
output_path = 'streams/input/'
output_archive = 'streams/archive'
tracker_path = 'logs/tracker.json'
col_labels = ["batch_id", "triggered", "start", "end"]

def clear_pushes():
    files = glob.glob(output_path+"*")
    for f in files:
        os.chmod(f, 0o777)
        os.remove(f)

    # shutil.rmtree(output_archive, ignore_errors=True)


def send_images(batch_id, images_to_push):
    json_dict = []
    for data, label in images_to_push:
        json_dict.append({"idx": batch_id, "data": data.tolist(), "label": label})
    log(batch_id)
    with open(output_path + f"batch_{batch_id}.json", "x") as outfile:
        json.dump(json_dict, outfile)

def log(batch_id):
    spark = SparkSession.builder. \
        appName("ImageDataPush"). \
        master("local[*]"). \
        config("spark.ui.port", "4051"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        getOrCreate()

    new_df = spark.createDataFrame([(batch_id, time.time(), 0.0, 0.0)], col_labels, schema)
    if not os.path.exists(tracker_path):
        new_df.toPandas().to_json(tracker_path, orient='records', force_ascii=False, lines=True)
    else:
        df = spark.read.json(tracker_path)
        df = df.unionByName(new_df)
        df.toPandas().to_json(tracker_path, orient='records', force_ascii=False, lines=True)

    spark.stop()

def push(rate, interval, stop):
    train_dataset, test_dataset, _ = createDataSet(dataset_path)

    complete_dataset = train_dataset
    complete_dataset.extend(test_dataset)

    size = len(complete_dataset)

    batch_id = 1
    while stop > 0:
        if batch_id > 1:
            time.sleep(interval)
        images_to_push = []
        for _ in range(rate):
            random_index = random.randint(1, size)
            images_to_push.append(complete_dataset[random_index])

        send_images(batch_id, images_to_push)
        print(f"Pushed {batch_id} data")
        batch_id += 1
        stop -= 1


if __name__ == '__main__':
    rate = 10
    interval = 10
    stop = 10
    clear_pushes()
    print("Starting pushing...")
    time.sleep(5)
    push(rate, interval, stop)
