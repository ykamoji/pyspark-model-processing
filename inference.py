from pyspark.sql import SparkSession
from transformers import ViTForImageClassification, ViTImageProcessor
from utils import createDataSet
import torch
import time
import numpy as np

dataset_path = '/Users/ykamoji/Documents/ImageDatabase/cifar-10-batches-py/'
pipeline_path = '/pipeline'

def startApp():
    spark = SparkSession.builder. \
        appName("ImageClassification"). \
        master("local[8]"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        getOrCreate()

    train_dataset, test_dataset, label_map = createDataSet(dataset_path)

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
        data = [float(x) for x in image.reshape(nChannels, height, width).transpose(1, 2, 0).flatten()]
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
        return class_label, 1 if gt == pred else 0

    def calc_acc(record):
        return record[0], f"{float(sum(record[1]) * 100 / len(record[1])):.2f}%"

    test = test_dataset[:5000]
    start = time.time()

    results = spark.sparkContext.parallelize(test)\
        .map(reshape_image)\
        .map(predictImage)\
        .groupByKey().mapValues(list).map(calc_acc).collect()

    print(f"Time taken = {(time.time() - start):.3f} sec \n\n")

    for class_label, acc in results:
        print(f"{class_label} : {acc}")

    spark.stop()


if __name__ == "__main__":
    startApp()
