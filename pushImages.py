from utils import createDataSet
import glob
import os
import random
import time
import json

dataset_path = '/Users/ykamoji/Documents/ImageDatabase/cifar-10-batches-py/'
output_path = 'streams/input/'


def clear_pushes():
    files = glob.glob(output_path+"*")
    for f in files:
        os.chmod(f, 0o777)
        os.remove(f)


def send_images(progress, images_to_push):
    json_dict = []
    for data, label in images_to_push:
        json_dict.append({"data": data.tolist(), "label": label})

    with open(output_path + f"data_{progress}.json", "x") as outfile:
        json.dump(json_dict, outfile)


def push(rate, interval, stop):
    train_dataset, test_dataset, _ = createDataSet(dataset_path)

    complete_dataset = train_dataset
    complete_dataset.extend(test_dataset)

    size = len(complete_dataset)

    progress = 0
    while stop > 0:
        images_to_push = []
        for _ in range(rate):
            random_index = random.randint(1, size)
            images_to_push.append(complete_dataset[random_index])

        send_images(progress, images_to_push)
        print(f"Pushed {progress} data")
        progress += 1
        stop -= 1
        time.sleep(interval)


if __name__ == '__main__':
    rate = 15
    interval = 2
    stop = 20
    clear_pushes()
    print("Starting pushing...")
    time.sleep(5)
    push(rate, interval, stop)
