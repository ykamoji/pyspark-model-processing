import torch
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from transformers import AutoModelForSequenceClassification
from utils import createTextDataset, collate_text_fn
from torch.utils.data import DataLoader

dataset_path = os.getcwd() + '/HateSpeechDataset.csv'

schema = StructType([
    StructField("model", StringType(), False),
    StructField("duration", FloatType(), False),
    StructField("total_texts", IntegerType() , False),
    StructField("batch_size", IntegerType(), False),
    StructField("average_batch_duration", FloatType(), False),
    StructField("accuracy", FloatType(), False)
])

col_labels = ["model", "duration", "total_texts", "batch_size", "average_batch_duration", "accuracy"]

train_dataset, test_dataset = createTextDataset(dataset_path)


def startInference(spark, model_name, device, total_texts, batch_size):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='models/')

    model.eval()
    device = torch.device(device)
    model.to(device)

    broadcasted_model = spark.sparkContext.broadcast(model)

    def preprocessText(record):
        try:
            # Directly extract inputs and labels from the record dictionary
            inputs = {key: val.to(device) for key, val in record.items() if key != 'labels'}
            labels = record['labels'].to(device)
            return inputs, labels
        except Exception as e:
            print("Error processing record:", e)
            print("Received record:", record)
            raise

    def predictText(record):
        inputs, labels = record
        with torch.no_grad():
            outputs = broadcasted_model.value(**inputs)
        logits = outputs.logits
        predictions = logits.argmax(-1)
        correct_predictions = (predictions == labels).float().mean().item()
        return correct_predictions

    test_dataloader = DataLoader(test_dataset[:total_texts], batch_size=batch_size, collate_fn=collate_text_fn, shuffle=True)

    start = time.time()
    accuracy = spark.sparkContext.parallelize(test_dataloader) \
        .map(preprocessText) \
        .map(predictText) \
        .mean()
    end = time.time() - start

    average_batch_duration = end / len(test_dataloader)
    df = spark.createDataFrame([(model_name, end, total_texts, batch_size, average_batch_duration, accuracy)],
                               col_labels, schema)

    return df


inference_params = {
    "huawei-noah/TinyBERT_General_4L_312D": [{"total_texts": 200, "batch_size": i} for i in [1, 2, 5, 10, 20, 50]]
}


def perform_profiling():

    test_dataloader = DataLoader(test_dataset[:2], batch_size=1, collate_fn=collate_text_fn)
    input = next(iter(test_dataloader))

    for model_name in ["microsoft/MiniLM-L12-H384-uncased", "huawei-noah/TinyBERT_General_4L_312D"]:
        for env in ["cpu"]: # Add "cuda" or "mps" if available
            with torch.autograd.profiler.profile(use_cuda=False) as prof:
                model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='models/')
                model.to(torch.device(env))
                input = {key: val.to(env) for key, val in input.items() if key != 'labels'}
                model(**input)
            print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=5))


if __name__ == "__main__":
    spark = SparkSession.builder. \
        appName("TextClassification"). \
        master("local[*]"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        getOrCreate()

    device = "cpu"
    dfs = None
    for model_name, paramList in inference_params.items():
        for params in paramList:
            df = startInference(spark, model_name, device, **params)
            dfs = df if dfs is None else dfs.union(df)

    dfs = dfs.withColumn("env", lit('mac_' + device))
    dfs.show(truncate=False)
    dfs.toPandas().to_csv("results/huaweibert.csv")

    spark.stop()

    perform_profiling()


