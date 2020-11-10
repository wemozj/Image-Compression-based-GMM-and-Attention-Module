#!/bin/bash
python train.py \
--config examples/example/config_low.json \
-n 2048_GMM_Attention_k4 \
--train /data/zhujun/IC_images/flicker_2W_images_PATCH256/ \
--val /home/zhujun/IMAGE_COMPRESSION/Baseline/compression_torch/images/