import torch
import os
import numpy as np
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.sql import SparkSession
from transformers import Trainer, TrainingArguments
from transformers.training_args import OptimizerNames
from datasets import load_dataset
import evaluate
from transformers import AutoImageProcessor, AutoModelForImageClassification

dataset_path = os.getcwd() + '/dataset'

conv_model_list = [
    "facebook/convnext-tiny-224",
    "facebook/convnext-small-224",
    "facebook/convnext-base-224",
    "facebook/convnext-large-224",
    "facebook/convnextv2-nano-22k-224",
    "facebook/convnextv2-tiny-22k-224",
    "facebook/convnextv2-base-22k-224",
    "facebook/convnextv2-large-22k-224",
    "facebook/convnextv2-huge-1k-224"
]

vit_model_list = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/deit-tiny-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/deit-base-patch16-224",
    "facebook/deit-tiny-distilled-patch16-224"
    "facebook/deit-base-distilled-patch16-224"
]

model_name = vit_model_list[2]

def get_fine_tuning_trainer_args(output_path):

    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        evaluation_strategy="steps",
        num_train_epochs=1,
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        learning_rate=0.01,
        warmup_ratio=0.1,
        warmup_steps=1,
        weight_decay=0,
        save_total_limit=2,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        optim=OptimizerNames.ADAMW_HF,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42,
        gradient_accumulation_steps=4,
    )

def build_metrics():

    _ = evaluate.load("accuracy", cache_dir="../metrics/", trust_remote_code=True)

    metric = evaluate.combine(["accuracy"])

    def compute_metrics(p):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1),
            references=p.label_ids
        )

    return compute_metrics

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


def startTraining(distributed_training=False):
    train_dataset = load_dataset('cifar10', split=f"train[:10]", verification_mode='no_checks',
                                 cache_dir=dataset_path)

    test_dataset = load_dataset('cifar10', split=f"test[:10]", verification_mode='no_checks',
                                cache_dir=dataset_path)

    pretrained_model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir='../models/')

    feature_extractor = AutoImageProcessor.from_pretrained(model_name, cache_dir='../models/')

    print(f"Starting fine-tuning on model {model_name}:")

    def preprocess(batchImage):
        inputs = feature_extractor(batchImage['img'], return_tensors='pt')
        inputs['label'] = batchImage['label']
        return inputs

    fine_tune_args = get_fine_tuning_trainer_args("../results/")

    def trainer():
        fine_tune_trainer = Trainer(
            model=pretrained_model,
            args=fine_tune_args,
            data_collator=collate_fn,
            compute_metrics=build_metrics(),
            train_dataset=train_dataset.with_transform(preprocess),
            eval_dataset=test_dataset.with_transform(preprocess),
        )
        print(f"Starting...")
        train_results = fine_tune_trainer.train()

        fine_tune_trainer.save_model(output_dir='results/models')

        fine_tune_trainer.log_metrics("train", train_results.metrics)
        fine_tune_trainer.save_state()

        metrics = fine_tune_trainer.evaluate(test_dataset.with_transform(preprocess))
        fine_tune_trainer.log_metrics("eval", metrics)
        fine_tune_trainer.save_metrics("eval", metrics)

        return train_results

    if distributed_training:
        import torch.distributed
        spark = SparkSession.builder. \
            appName("ImageClassification"). \
            master("local[*]"). \
            config("spark.executor.memory", "16G"). \
            config("spark.driver.memory", "16G"). \
            config("spark.executor.resource.gpu.amount", "1"). \
            config("spark.driver.resource.gpu.amount", "1"). \
            config("spark.task.resource.gpu.amount", "1"). \
            config("spark.executor.resource.gpu.discoveryScript", os.getcwd() + "/discovery.sh"). \
            config("spark.driver.resource.gpu.discoveryScript", os.getcwd() + "/discovery.sh"). \
            getOrCreate()
        torch.distributed.init_process_group(backend="nccl")
        results = TorchDistributor(local_mode=True, use_gpu=False).run(trainer)
        torch.distributed.destroy_process_group()
        spark.stop()

    else:
        trainer()


if __name__ == "__main__":
    startTraining(True)