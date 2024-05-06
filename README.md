#### Requirements
```angular2html
torch==2.2.0
torchmetrics==1.3.1
torchvision==0.17.0
transformers==4.38.2
evaluate==0.4.1
numpy==1.24.4
re==2.2.1
tensorboard==2.16.2
json==2.0.9
seaborn==0.12.2
matplotlib==3.7.1
pandas==1.5.3
PIL==10.0.1
```

#### Image Classification Task:
```angular2html
Training: image_training.py
Inference: image_inference.py
Analysis: analyse.py
```

#### Text Classification Task:
```angular2html
Training: text_training.py
Inference: text_inference.py
Analysis: Included in text_training.py and text_inference.py
```

#### Streaming Process:
```angular2html
Push Images into the stream: pushImages.py
Streaming: streaming.py
Analysis: analyse.py 
```

In the terminal:

#### To start the streaming process
```angular2html
python3 -u pushImages.py > logs/trigger.log 2>&1 &
python3 -u streaming.py > logs/streaming.log 2>&1 &
```

#### To end the streaming process
```angular2html
ps -e | grep 'pushImages.py' | grep -v grep | awk '{print $1}' | xargs kill
ps -e | grep 'streaming.py' | grep -v grep | awk '{print $1}' | xargs kill
```

#### Datasets:
	Images: CIFAR-10 downloaded automatically with image_training.py
	Text: Make sure to include HateSpeechDataset.csv in your directory.


#### Miscellaneous:
utils.py - Contains various utility functions in regards to ETL functions for the datasets

