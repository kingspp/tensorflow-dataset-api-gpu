#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/Users/prathyushsp/Git/tensorflow-dataset-api-gpu:/home/prathyush.sp/Git/tensorflow-dataset-api-gpu

echo "##################################################################################################"
echo "###           Running FFN Experiments                                                           ##"
echo "##################################################################################################"


python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_1.json normal | tee mnist_ffn_config_gpu_1_1_normal.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_1.json profiled | tee mnist_ffn_config_gpu_1_1_profiled.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_1.json separate | tee mnist_ffn_config_gpu_1_1_separate.log
python3.6 -u ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_1_1.json | tee mnist_ffn_config_gpu_1_1_report.log

python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_2.json normal | tee mnist_ffn_config_gpu_1_2_normal.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_2.json profiled | tee mnist_ffn_config_gpu_1_2_profiled.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_2.json separate | tee mnist_ffn_config_gpu_1_2_separate.log
python3.6 -u ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_1_2.json | tee mnist_ffn_config_gpu_1_2_report.log

python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_3.json normal | tee mnist_ffn_config_gpu_1_3_normal.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_3.json profiled | tee mnist_ffn_config_gpu_1_3_profiled.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_3.json separate | tee mnist_ffn_config_gpu_1_3_separate.log
python3.6 -u ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_1_3.json | tee mnist_ffn_config_gpu_1_3_report.log

python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_4.json normal | tee mnist_ffn_config_gpu_1_4_normal.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_4.json profiled | tee mnist_ffn_config_gpu_1_4_profiled.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_4.json separate | tee mnist_ffn_config_gpu_1_4_separate.log
python3.6 -u ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_1_4.json | tee mnist_ffn_config_gpu_1_4_report.log

python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_5.json normal | tee mnist_ffn_config_gpu_1_5_normal.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_5.json profiled | tee mnist_ffn_config_gpu_1_5_profiled.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_5.json separate | tee mnist_ffn_config_gpu_1_5_separate.log
python3.6 -u ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_1_5.json | tee mnist_ffn_config_gpu_1_5_report.log

python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_6.json normal | tee mnist_ffn_config_gpu_1_6_normal.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_6.json profiled | tee mnist_ffn_config_gpu_1_6_profiled.log
python3.6 -u ./../../benchmark/template.py ./mnist_ffn_config_gpu_1_6.json separate | tee mnist_ffn_config_gpu_1_6_separate.log
python3.6 -u ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_1_6.json | tee mnist_ffn_config_gpu_1_6_report.log


