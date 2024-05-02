#!/bin/bash

ps -e | grep 'pushImages.py' | grep -v grep | awk '{print $1}' | xargs kill
ps -e | grep 'streaming.py' | grep -v grep | awk '{print $1}' | xargs kill