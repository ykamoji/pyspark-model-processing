#!/bin/bash

python3 -u pushImages.py > logs/trigger.log 2>&1 &
python3 -u streaming.py > logs/streaming.log 2>&1 &
