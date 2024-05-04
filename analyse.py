import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode, col, desc, when, array, struct, concat
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType, StringType
from utils import model_details, select_cast
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


def to_explode(df, by):
    cols, dtypes = zip(*((c, t) for (c, t) in df.dtypes if c not in by))
    kvs = explode(array([
        struct(lit(c).alias("CATEGORY"), col(c).alias("Normalized_value")) for c in cols
    ])).alias("kvs")
    return df.select(by + [kvs]).select(by + ["kvs.CATEGORY", "kvs.Normalized_value"])


def analyse_training_results(spark):

    train_results_df = spark.read.option("header", True).csv("results/training_results.csv")

    train_results_df = train_results_df\
        .withColumn("Model", when(col("Model") == "google/vit-base-patch16-224", "ViT")\
                    .otherwise(train_results_df.model))\
        .withColumn("Model", when(col("Model") == "facebook/deit-tiny-patch16-224", "DeiT") \
                    .otherwise(col("Model")))\
        .withColumn("Model", when(col("Model") == "facebook/convnext-small-224", "ConvNext") \
                    .otherwise(col("Model")))\
        .withColumn("Model", when(col("Model") == "facebook/convnextv2-nano-22k-224", "ConvNextV2") \
                    .otherwise(col("Model")))\
        .filter((col("Model") == "ViT") | (col("Model") == "DeiT") | (col("Model") == "ConvNext") |
                (col("Model") == "ConvNextV2"))\
        .withColumn("env", when(col("env") == 'COLAB_GPU', "LINUX_GPU")\
                    .otherwise(when(col("env") == 'GPU', "MAC_GPU").otherwise(col("env"))))\
        .select(col("env"), col("Model"), col("train_runtime").cast(FloatType()), col("total_flos"), 
                col("train_steps_per_second"),col("train_samples_per_second"))

    for column in ["train_runtime", "total_flos", "train_steps_per_second", "train_samples_per_second"]:
        column_max = train_results_df.agg({column: "max"}).collect()[0][0]
        train_results_df = train_results_df.withColumn(column, col(column) / column_max)

    train_results_df = train_results_df.select(col("env"), col("Model"), col("total_flos").alias("FLOS"),
                                               col("train_runtime").alias("Runtime"),
                                               col("train_samples_per_second").alias("Batch_per_second"),
                                               col("train_steps_per_second").alias("Step_per_second"))

    # train_results_df.show(truncate=False)

    train_results_df = to_explode(train_results_df, ['Model', "env"])

    # train_results_df.show(100, truncate=False)

    for env in ["MAC_GPU", "LINUX_GPU"]:
        plt.style.use('seaborn-v0_8-darkgrid')
        ax = sns.barplot(train_results_df.filter(col("env") == env).toPandas(), x="CATEGORY", y="Normalized_value",
                         hue="Model", errorbar=None)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(0.1, 1))
        plt.title(f"{env} Training")
        # plt.yticks([])
        # plt.savefig(f"results/{env}_Training")
        plt.show()


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

    # validate_baseline_batch_size(df_batch_size_comparison)

    # train_results_df.show(100, truncate=False)

    individual_model_plots(df_batch_size_comparison)

    df_test_comparison = spark.read.option("header", True) \
        .schema(inference_schema).csv("results/inference_results_test_comparison.csv")

    # df_test_comparison.show(truncate=False)

    # validate_baseline_test(df_test_comparison)

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
        groupByKey().map(list). \
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


def individual_model_plots(df):

    df = df \
        .withColumn("Model", when(col("Model") == "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", "ViT") \
                    .otherwise(df.model)) \
        .withColumn("Model", when(col("Model") == "tzhao3/DeiT-CIFAR10", "DeiT") \
                    .otherwise(col("Model"))) \
        .withColumn("Model", when(col("Model") == "ahsanjavid/convnext-tiny-finetuned-cifar10", "ConvNext") \
                    .otherwise(col("Model"))) \
        .withColumn("Model", when(col("Model") == "facebook/convnextv2-tiny-22k-224", "ConvNextV2") \
                    .otherwise(col("Model"))) \
        .filter((col("Model") == "ViT") | (col("Model") == "DeiT") | (col("Model") == "ConvNext") |
                (col("Model") == "ConvNextV2")) \
        .withColumn("env", when(col("env") == 'cpu', "LINUX_CPU") \
                    .otherwise(when(col("env") == 'mac_cpu', "MAC_CPU")
                               .otherwise(when(col("env") == 'cuda', "LINUX_CUDA")
                                          .otherwise(col("env"))))) \
        .select(col("Model"), col("batch_size").alias("Batch_size"),
                col("average_batch_duration").alias("Average_batch_duration (Sec)"), col("env").alias("Env"))

    # df.show(400, truncate=False)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    ax = sns.lineplot(data=df.toPandas(), x="Batch_size", y="Average_batch_duration (Sec)", hue="Model", style="Env", ax=ax)
    sns.move_legend(ax, "upper right", bbox_to_anchor=(0.2, 1))
    # plt.savefig(f"results/Average_batch_duration")
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

    vit_model_df = df.select(col("duration").alias("Duration (Sec)"), col("total_images").alias("Test_size"),
                             col("env").alias("Env"))\
                    .withColumn("Env", when(col("env") == 'cpu', "LINUX_CPU")\
                    .otherwise(when(col("Env") == 'mac_cpu', "MAC_CPU")
                               .otherwise(when(col("Env") == 'cuda', "LINUX_CUDA")
                               .otherwise(col("Env")))))

    plt.style.use('seaborn-v0_8-darkgrid')
    sns.lineplot(data=vit_model_df.toPandas(), x="Test_size", y="Duration (Sec)", hue='Env', style="Env")
    plt.title("ViT Model")
    # plt.savefig(f"results/Test_size_duration")
    plt.show()


#################################### Streaming analysis ####################################


def streaming_analysis(spark):

    files = glob.glob("results/streaming/speed_5_batch_5/*/*.json")
    dfs = None
    for tracker in files:
        details = tracker.split("/")[-2]
        model = details.split('_')[0]
        env = "MAC_CPU" if details.split('_')[1] == "cpu" else "LINUX_GPU"
        df = spark.read.json(tracker)
        df = df.select(col("batch_id"), col("batch_size"), col("accuracy"),
                col("triggered"),select_cast('triggered').alias('triggered_time'),
                col("start"), select_cast('start').alias('start_time'),
                col("end"), select_cast('end').alias('end_time')).\
                withColumn("Model", lit(model)).withColumn("Env", lit(env))

        if not dfs:
            dfs = df
        else:
            dfs = dfs.union(df)

    # dfs.show(500, truncate=False)

    dfs = dfs.withColumn("Latency", col("start") - col("triggered"))\
        .withColumn("Throughput", col("batch_size") / (col("end") - col("start")))\
        .select(col("Model"), col("Env"), col("batch_size").alias("Images"), col("accuracy"), col("triggered_time"),
                col("start_time"), col("end_time"), col("Latency"), col("Throughput")).orderBy(col("Model"))

    # dfs.show(500, truncate=False)
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.lineplot(data=dfs.toPandas(), x="Images", y="Latency", hue="Model", style="Env")
    # plt.savefig(f"results/Latency")
    plt.show()
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.lineplot(data=dfs.toPandas(), x="Images", y="Throughput", hue="Model", style="Env")
    # plt.savefig(f"results/Throughput")
    plt.show()


if __name__ == "__main__":
    spark = SparkSession.builder. \
        appName("Analysis"). \
        master("local[*]"). \
        config("spark.executor.memory", "16G"). \
        config("spark.driver.memory", "16G"). \
        getOrCreate()

    # collect_training_results(spark)

    analyse_training_results(spark)

    analyse_inference(spark)

    streaming_analysis(spark)

    spark.stop()
