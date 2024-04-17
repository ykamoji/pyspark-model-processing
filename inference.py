import torch
import numpy as np
import time
from pyspark.sql import SparkSession
from transformers import ViTForImageClassification, ViTImageProcessor
from utils import createDataSet, collate_fn
from torch.utils.data import DataLoader

dataset_path = '/Users/ykamoji/Documents/ImageDatabase/cifar-10-batches-py/'

def startApp():

    train_dataset, test_dataset, label_map = createDataSet(dataset_path)

    processor = ViTImageProcessor.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',
                                                  cache_dir='models/')
    model = ViTForImageClassification.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',
                                                      cache_dir='models/')
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    def preprocessImage(record):
        images, labels = record[0], record[1]
        height = 32
        width = 32
        nChannels = 3
        # data = [float(x) for x in images.reshape(nChannels, height, width).transpose(1, 2, 0).flatten()]
        data = torch.stack([images[i].reshape(nChannels, height, width).permute(1, 2, 0) for i in range(images.shape[0])])
        # print(data.shape, labels.shape)
        return data, labels

    def predictImage(record):
        gts = record[1]
        class_labels = [label_map[gt.item()] for gt in gts]
        # image = np.array(record[0]).reshape(32, 32, 3).astype(np.uint8)
        images = record[0]
        input = processor(images=images, return_tensors="pt")
        input = input.to(device)
        gts.to(device)
        input = collate_fn(input, gts)
        output = model(**input)
        # print(output)
        logits = output.logits.cpu()
        predictions = (gts == logits.argmax(-1)).detach().cpu().numpy()
        return class_labels, predictions

    def processBatchPredictions(record):
        class_labels, predictions = record[0], record[1]
        iterables = []
        for i in range(len(predictions)):
            iterables.append((class_labels[i], predictions[i]))
        return iter(iterables)

    def calc_acc(record):
        return record[0], f"{float(sum(record[1]) * 100 / len(record[1])):.2f}%"

    test = test_dataset[:500]

    test_dataloader = DataLoader(test, batch_size=10, shuffle=True)

    spark = SparkSession.builder. \
        appName("ImageClassification"). \
        master("local[8]"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        getOrCreate()

    start = time.time()

    results = spark.sparkContext.parallelize(test_dataloader)\
        .map(preprocessImage)\
        .map(predictImage)\
        .flatMap(processBatchPredictions)\
        .groupByKey().mapValues(list).map(calc_acc).collect()

    print(f"Time taken = {(time.time() - start):.3f} sec \n\n")

    for class_label, acc in results:
        print(f"{class_label} : {acc}")

    spark.stop()


if __name__ == "__main__":
    startApp()
