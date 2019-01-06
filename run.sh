#!/usr/bin/env bash

python3 benchmark/template.py ./benchmark/configs/sample_config.json normal
python3 benchmark/template.py ./benchmark/configs/sample_config.json profiled
python3 benchmark/template.py ./benchmark/configs/sample_config.json separate
