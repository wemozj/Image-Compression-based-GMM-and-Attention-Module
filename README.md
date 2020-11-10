
## Overview

## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)

## Prerequisites

You should install the libraries of this repo.

```
pip install -r requirements.txt
```

## Data Preparation

We need to first prepare the training and validation data.
The trainging data is from flicker.com.
You can obtain the training data according to description of [CompressionData](https://github.com/liujiaheng/CompressionData).

The validation data is the popular kodak dataset.
```
bash data/download_kodak.sh
```

## Training 

For high bitrate (4096, 6144, 8192), the out_channel_N is 192 and the out_channel_M is 320 in 'config_high.json'.
For low bitrate (256, 512, 1024, 2048), the out_channel_N is 128 and the out_channel_M is 192 in 'config_low.json'.

