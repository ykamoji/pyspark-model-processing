import pickle


def createDataSet(dataset_path):
    train_dataset = []
    test_dataset = []
    for i in range(1, 2):
        data = pickle.load(open(dataset_path + f'data_batch_{i}', 'rb'), encoding='latin-1')
        train_dataset.extend(zip(data["data"], data["labels"]))

    test_data = pickle.load(open(dataset_path + f'test_batch', 'rb'), encoding='latin-1')
    test_dataset.extend(zip(test_data["data"], test_data["labels"]))

    label_map = get_label_map(dataset_path)

    return train_dataset, test_dataset, label_map


def get_label_map(dataset_path):
    meta = pickle.load(open(dataset_path + f'batches.meta', 'rb'), encoding='latin-1')
    return {index: label for index, label in enumerate(meta['label_names'])}
