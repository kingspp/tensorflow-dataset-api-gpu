#!/usr/bin/env bash

python3.6 benchmark/template.py ./benchmark/configs/sample_config.json normal
python3.6 benchmark/template.py ./benchmark/configs/sample_config.json profiled
python3.6 benchmark/template.py ./benchmark/configs/sample_config.json separate
python3.6 ./benchmark/report_generator.py ./benchmark/configs/sample_config.json