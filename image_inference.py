import os
import torch
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from transformers import AutoModelForImageClassification, AutoImageProcessor
from utils import createImageDataSet, collate_image_fn
from torch.utils.data import DataLoader

## To download the CIFAR10 dataset, run below two commands
# !wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# !tar -xvzf cifar-10-python.tar.gz
# Update the below path after downloading the files
dataset_path = os.getcwd() + '/cifar-10-batches-py/'


schema = StructType([
    StructField("model", StringType(), False),
    StructField("duration", FloatType(), False),
    StructField("total_images", IntegerType() , False),
    StructField("batch_size", IntegerType(), False),
    StructField("average_batch_duration", FloatType(), False),
    StructField("balanced_accuracy", FloatType(), False)
])

col_labels = ["model","duration","total_images","batch_size","average_batch_duration","balanced_accuracy"]

train_dataset, test_dataset, label_map = createImageDataSet(dataset_path)


def startInference(spark, model_name, device, total_images, batch_size):
    """
    Creates a spark cluster to parallelize the dataset, predictions, and calculating accuracies using RDD.
    :param spark:
    :param model_name:
    :param device: Environment cpu or cuda
    :param total_images:
    :param batch_size:
    :return: dataframe containing inference results
    """

    if "results" in model_name:
        parent_model = "/".join(model_name.split('/')[1:-1])
        processor = AutoImageProcessor.from_pretrained(parent_model, cache_dir='models/')
        model = AutoModelForImageClassification.from_pretrained(model_name, use_safetensors=True)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir='models/')
        model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir='models/')

    model.eval()
    device = torch.device(device)
    model.to(device)

    broadcasted_model = spark.sparkContext.broadcast(model)

    def preprocessImage(record):
        """
        Preprocesses the images to compatible format required by the model.
        :param record:
        :return: data, labels
        """
        images, labels = record[0], record[1]
        height = 32
        width = 32
        nChannels = 3
        # data = [float(x) for x in images.reshape(nChannels, height, width).transpose(1, 2, 0).flatten()]
        data = torch.stack(
            [images[i].reshape(nChannels, height, width).permute(1, 2, 0) for i in range(images.shape[0])])
        # print(data.shape, labels.shape)
        return data, labels

    def predictImage(record):
        """
        Get predictions of the image classes bases on the ground truth labels and duration.
        Load input to the device: cpu or cuda
        :param record:
        :return: class labels, predictions, duration
        """
        gts = record[1]
        class_labels = [label_map[gt.item()] for gt in gts]
        # image = np.array(record[0]).reshape(32, 32, 3).astype(np.uint8)
        images = record[0]
        input = processor(images=images, return_tensors="pt")
        input = input.to(device)
        input = collate_image_fn(input, gts)
        start = time.time()
        output = broadcasted_model.value(**input)
        end = time.time()
        # print(output)
        logits = output.logits.cpu()
        predictions = (gts == logits.argmax(-1)).detach().cpu().numpy()
        return class_labels, predictions, end - start

    def processBatchPredictions(record):
        """
        Flattening of the list of records based on the class label
        :param record:
        :return: iterable of class label, predictions and duration
        """
        class_labels, predictions, batch_duration = record
        iterables = []
        for i in range(len(predictions)):
            iterables.append((class_labels[i], (predictions[i], batch_duration)))
        return iter(iterables)

    def calc_acc(record):
        """
        Calculate the accuracies and average batch duration for the class label.
        :param record: class label, predictions and duration
        :return: class label, accuracy and duration
        """
        class_label = record[0]
        predictions = float(sum(details[0] for details in record[1]) * 100 / len(record[1]))
        average_batch_duration = float(sum(details[1] for details in record[1]) / len(record[1]))
        return class_label, predictions, average_batch_duration

    test = test_dataset[:total_images]

    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    start = time.time()

    # The inference is done by parallelizing the test datasets using PySpark RDD.
    results = spark.sparkContext.parallelize(test_dataloader) \
        .map(preprocessImage) \
        .map(predictImage) \
        .flatMap(processBatchPredictions) \
        .groupByKey().mapValues(list).map(calc_acc).collect()

    end = round(time.time() - start, 3)

    # print(f"Time taken = {end:.3f} sec\n")
    balanced_accuracy = 0
    average_batch_duration = 0
    for class_label, acc, avg_batch in results:
        # print(f"{class_label} : {acc:.3f}% (Avg: {avg_batch:.3f})")
        balanced_accuracy += acc
        average_batch_duration += avg_batch

    balanced_accuracy = round(balanced_accuracy/len(results),3)
    average_batch_duration = round(average_batch_duration/len(results),5)

    if "results/" in model_name:
        model_name = "/".join(model_name.split('/')[1:-1])

    df = spark.createDataFrame([(model_name, end, total_images, batch_size, average_batch_duration, balanced_accuracy)],
                               col_labels, schema)

    df.show(truncate=False)

    return df


common_params = [{"total_texts": 200, "batch_size": i} for i in [1, 2, 5, 7, 10, 20, 30, 40]]

# common_params = [{"total_texts": 200, "batch_size": 20} for i in
#                  [20, 40, 80, 100, 120, 160, 200, 300, 400, 500, 700, 1000, 2000, 3000, 4000, 5000, 7000, 1000]]

inference_params = {
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10":common_params,
    "tzhao3/DeiT-CIFAR10": common_params,
    "ahsanjavid/convnext-tiny-finetuned-cifar10":common_params,
    "results/facebook/deit-tiny-patch16-224/models":common_params,
    "results/facebook/deit-tiny-distilled-patch16-224/models":common_params,
    "results/facebook/convnextv2-tiny-22k-224/models":common_params
}


def perform_inference():
    """
    Performs distributed data evaluations of the model with the given parameters.
    :return:
    """
    spark = SparkSession.builder. \
        appName("ImageClassification"). \
        master("local[*]"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        getOrCreate()

    device = "cpu"
    dfs = None
    for model_name, paramList in inference_params.items():
        for params in paramList:
            params = {**params, **{"device": device}}
            df = startInference(spark, model_name, **params)
            if not dfs:
                dfs = df
            else:
                dfs = dfs.union(df)

    dfs = dfs.withColumn("env", lit('mac_' + device))

    dfs.show(truncate=False)

    dfs.toPandas().to_csv("results/inference_results.csv")

    spark.stop()


def perform_profiling():
    """
    Profiling the models for the different environments.
    :return:
    """

    for model_name in ["aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", "ahsanjavid/convnext-tiny-finetuned-cifar10"]:
        for env in ["mps", "cpu"]:
            with torch.autograd.profiler.profile(use_cuda=False) as prof:
                processor = AutoImageProcessor.from_pretrained(model_name, cache_dir='models/')
                model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir='models/')
                input = processor(test_dataset[0][0].reshape(1,3,32,32), return_tensors="pt")
                model.to(torch.device(env))
                input = input.to(torch.device(env))
                model(**input)
            print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=5))


if __name__ == "__main__":

    perform_inference()

    perform_profiling()





