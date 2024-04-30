import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode, col, desc
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType

## Training analysis

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

training_logs = glob.glob("results/*/*/*/trainer_state.json", recursive=True)

log_history = StructType([
    StructField("loss", FloatType(), True),
    StructField("step", FloatType(), False),
    StructField("eval_loss", FloatType(), True),
    StructField("eval_accuracy", FloatType(), True),
    StructField("eval_f1", FloatType(), True),
    StructField("eval_precision", FloatType(), True),
    StructField("eval_recall", FloatType(), True),
    StructField("eval_runtime", FloatType(), True),
    StructField("eval_samples_per_second", FloatType(), True),
    StructField("eval_steps_per_second", FloatType(), True),
    StructField("total_flos", FloatType(), True),
    StructField("train_loss", FloatType(), True),
    StructField("train_runtime", FloatType(), True),
    StructField("train_samples_per_second", FloatType(), True),
    StructField("train_steps_per_second", FloatType(), True),
])

schema = StructType([
    StructField("total_flos", FloatType(), False),
    StructField("train_batch_size", IntegerType(), False),
    StructField("global_step", IntegerType(), False),
    StructField("log_history", ArrayType(log_history), False),
])

spark = SparkSession.builder. \
    appName("Analysis"). \
    master("local[*]"). \
    config("spark.executor.memory", "16G"). \
    config("spark.driver.memory", "16G"). \
    getOrCreate()

dfs = None

for file in training_logs:
    model_group = file.split("/")[1] + '/'
    model_name = file.split("/")[2]
    env = "GPU"
    if ":" in model_name:
        env = model_name.split(':')[1]
        model_name = model_group + model_name.split(':')[0]
    else:
        model_name = model_group + model_name

    df = spark.read.json(file, schema=schema, multiLine=True)

    df_logs = df.select(explode("log_history").alias("log"))

    df_train_results = df_logs.filter(col("log.train_runtime").isNotNull()). \
        withColumn("model", lit(model_name)). \
        select("model", "log.train_runtime", "log.train_samples_per_second", "log.train_steps_per_second")

    df_eval_results = df_logs.filter(col("log.eval_accuracy").isNotNull()). \
        orderBy(desc(col("log.step"))).limit(1).withColumn("model", lit(model_name)). \
        select("model", "log.eval_accuracy", "log.eval_f1", "log.eval_loss", "log.eval_precision", "log.eval_recall",
               "log.eval_runtime", "log.eval_samples_per_second", "log.eval_steps_per_second")

    details = model_details[model_name]

    df_model = df.withColumn("model", lit(model_name)). \
        withColumn("env", lit(env)). \
        withColumn("params", lit(str(details["params"]) + "M")). \
        withColumn("size", lit(str(details["size"]) + "MB")). \
        select("model", "env", "params", "size", "train_batch_size", "global_step", "total_flos")

    final_df = df_model.join(df_train_results, on="model").join(df_eval_results, on="model")

    if not dfs:
        dfs = final_df
    else:
        dfs = dfs.union(final_df)

dfs.orderBy(col("model")).show(truncate=False)

# 3. Plot the graphs from DFs using pandas / sns / matlab

dfs.toPandas().to_csv("results/training_results.csv")

spark.stop()




## Inference analysis

# -1. Add more batches for colab / CPU / Mac comparisons (done) and add few more models after finetuning on cifar 10

# 0. Add one model for GPU vs CPU comparison with more images

# 1. Collect all inferences into 1 DF.

# 2. Plot the graphs from DFs using pandas / sns / matlab


## Steaming analysis
