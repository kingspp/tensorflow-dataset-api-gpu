#!/usr/bin/env bash

python3.6 benchmark/template.py ./benchmark/configs/mnist_cnn_config.json normal
python3.6 benchmark/template.py ./benchmark/configs/mnist_cnn_config.json profiled
python3.6 benchmark/template.py ./benchmark/configs/mnist_cnn_config.json separate
python3.6 ./benchmark/report_generator.py ./benchmark/configs/mnist_cnn_config.json