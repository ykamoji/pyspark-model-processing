import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode, col, desc
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType
from utils import model_details


## Training analysis
def collect_training_results(spark):
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

    dfs.toPandas().to_csv("results/training_results.csv")

    ##TODO:: Plot the graphs from DFs using sns

## Inference analysis
def analyse_inference(spark):
    ##TODO:: Plot the graphs from DFs using pandas & sns
    pass


## Steaming analysis
def streaming_analysis(spark):
    pass
    ##TODO:: Plot the graphs from DFs using pandas & sns

if __name__ == "__main__":

    spark = SparkSession.builder. \
        appName("Analysis"). \
        master("local[*]"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        getOrCreate()

    # collect_training_results(spark)

    analyse_inference(spark)

    streaming_analysis(spark)

    spark.stop()