import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode, col, desc
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType, StringType
from utils import model_details
import matplotlib.pyplot as plt
import seaborn as sns


#################################### Training analysis ####################################
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
            select("model", "log.eval_accuracy", "log.eval_f1", "log.eval_loss", "log.eval_precision",
                   "log.eval_recall",
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


#################################### Inference analysis ####################################
def analyse_inference(spark):
    inference_schema = StructType([
        StructField("model", StringType(), False),
        StructField("duration", FloatType(), False),
        StructField("total_images", FloatType(), False),
        StructField("batch_size", FloatType(), False),
        StructField("average_batch_duration", FloatType(), False),
        StructField("balanced_accuracy", FloatType(), False),
        StructField("env", StringType(), False),
    ])

    df_batch_size_comparison = spark.read.option("header", True) \
        .schema(inference_schema).csv("results/inference_results_batch_size_comparison.csv")

    # df_batch_size_comparison.show(truncate=False)

    model_keys = validate_baseline_batch_size(df_batch_size_comparison)

    individual_model_plots(model_keys[3], df_batch_size_comparison)

    df_test_comparison = spark.read.option("header", True) \
        .schema(inference_schema).csv("results/inference_results_test_comparison.csv")

    # df_test_comparison.show(truncate=False)

    validate_baseline_test(df_test_comparison)

    test_plots(df_test_comparison)


def validate_baseline_batch_size(df_batch_size_comparison):

    df_rdd = df_batch_size_comparison.rdd

    def model_env_acc_map(record):
        model, acc, env = record[0], record[-2], record[-1]
        return model, (env, round(acc, 3))

    def append(a, b):
        a += b
        return a

    def extend(a, b):
        a.extend(b)
        return a

    check_correct_accuracies = df_rdd.map(model_env_acc_map). \
        groupByKey().mapValues(list). \
        combineByKey(lambda k: list(set(k)), append, extend)

    detail = StructType([
        StructField("environment", StringType(), False),
        StructField("accuracy", StringType(), False),
    ])
    schema = StructType([
        StructField("model_name", StringType(), False),
        StructField("detail", ArrayType(detail), False),
    ])
    check_df = check_correct_accuracies.toDF(schema=schema).withColumn("detail", explode("detail").alias("detail")) \
        .select("model_name", "detail.environment", "detail.accuracy")

    check_df.show(truncate=False)

    model_keys = check_correct_accuracies.keys().collect()
    return model_keys


def individual_model_plots(model, df):
    print(f"Plotting for {model}")

    model_df = df.filter(col("model") == model).\
        select("duration", "batch_size", "average_batch_duration", "env")

    model_df.show(truncate=False)

    for compare in ["duration", "average_batch_duration"]:
        sns.lineplot(data=model_df.toPandas(), x="batch_size", y=compare, hue='env', palette="flare")
        plt.show()
        sns.barplot(data=model_df.toPandas(), x="batch_size", y=compare, hue="env", estimator="sum",
                    errorbar=None)
        plt.show()


def validate_baseline_test(df_test_comparison):

    df_rdd = df_test_comparison.rdd

    def test_env_acc_map(record):
        test_size, acc, env = record[2], record[-2], record[-1]
        return test_size, (env, round(acc, 3))

    def append(a, b):
        a += b
        return a

    def extend(a, b):
        a.extend(b)
        return a

    check_correct_accuracies = df_rdd.map(test_env_acc_map). \
        groupByKey().mapValues(list). \
        combineByKey(lambda k: list(set(k)), append, extend)

    detail = StructType([
        StructField("environment", StringType(), False),
        StructField("accuracy", StringType(), False),
    ])
    schema = StructType([
        StructField("test_size", FloatType(), False),
        StructField("detail", ArrayType(detail), False),
    ])
    check_df = check_correct_accuracies.toDF(schema=schema).withColumn("detail", explode("detail").alias("detail")) \
        .select("test_size","detail.environment", "detail.accuracy")

    check_df.show(100, truncate=False)


def test_plots(df):

    vit_model_df = df.select("duration", "total_images", "average_batch_duration", "env")

    for compare in ["duration", "average_batch_duration"]:
        sns.lineplot(data=vit_model_df.toPandas(), x="total_images", y=compare, hue='env', palette="flare")
        plt.show()
        sns.barplot(data=vit_model_df.toPandas(), x="total_images", y=compare, hue="env", estimator="sum",
                    errorbar=None)
        plt.show()


#################################### Streaming analysis ####################################


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

    # streaming_analysis(spark)

    spark.stop()
