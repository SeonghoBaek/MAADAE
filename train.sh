#!/bin/bash

python3 model.py --mode train --model_path model/m.ckpt --train_data data/train/cat --img_size 32 --epoch 1000 --batch_size 8 
