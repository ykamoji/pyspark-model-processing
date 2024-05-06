import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Creating datasets compatible with Hugging Face Trainer
from torch.utils.data import Dataset

dataset_path = os.getcwd() + "/HateSpeechDataset.csv"


def get_fine_tuning_trainer_args(output_path):
    return TrainingArguments(
        output_dir=output_path + '/training/',
        logging_dir=output_path + '/logs/',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=3,
        save_steps=40,
        eval_steps=40,
        logging_steps=10,
        learning_rate=5.e-05,
        warmup_ratio=0.1,
        warmup_steps=1,
        weight_decay=0,
        save_total_limit=2,
        metric_for_best_model='accuracy',  # Ensure this is set to a single metric
        greater_is_better=True,
        optim='adamw_hf',
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42,
        gradient_accumulation_steps=4,
    )


def build_metrics():
    metric_combined = ["accuracy", "precision", "recall", "f1"]
    metric_collector = []
    for evals in metric_combined:
        metric_collector.append(evaluate.load(evals, cache_dir="metrics/", trust_remote_code=True))

    def compute_metrics(p):
        logits, labels = p.predictions, p.label_ids
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    return compute_metrics


def preprocess(dataframe, tokenizer):
    # Check unique values in the 'Label' column
    unique_labels = dataframe['Label'].unique()
    print("Unique labels:", unique_labels)

    # Convert labels to int, handling non-integer values safely
    def safe_int_convert(x):
        try:
            return int(x)
        except ValueError:
            return None  # or use a default value or raise an informative error

    dataframe['Label'] = dataframe['Label'].apply(safe_int_convert)
    dataframe = dataframe.dropna(subset=['Label'])  # Remove rows with invalid labels

    # Tokenize the content
    inputs = tokenizer(dataframe['Content'].tolist(), truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    labels = dataframe['Label'].tolist()
    
    return inputs, labels


def skip_rows(index):
    if index == 0:
        return False  # Ensure the header is not skipped
    return np.random.rand() > 0.05  # Skip rows with 95% probability


def startTraining(model_name):

    data = pd.read_csv(dataset_path, skiprows=lambda x: skip_rows(x))
    data = data.head(1000)
    data = data.sample(frac=1).reset_index(drop=True)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='models/')
    train_inputs, train_labels = preprocess(train_data, tokenizer)
    test_inputs, test_labels = preprocess(test_data, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='models/')
    args = get_fine_tuning_trainer_args(f"/results/{model_name}")
    
    class CustomDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings  # Encodings should be a dict of tensors from the tokenizer
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: self.encodings[key][idx].clone().detach() for key in self.encodings}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_inputs, train_labels)
    test_dataset = CustomDataset(test_inputs, test_labels)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=build_metrics()  # This should reference your function
    )

    train_result = trainer.train()
    trainer.save_model(f'/results/{model_name}/model')
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    # startTraining("microsoft/MiniLM-L12-H384-uncased")
    startTraining("huawei-noah/TinyBERT_General_4L_312D")
    # startTraining("distilbert/distilbert-base-uncased")