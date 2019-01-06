#!/usr/bin/env bash
echo "##################################################################################################"
echo "###           Running FFN Experiments                                                           ##"
echo "##################################################################################################"

python3.6 ./../../benchmark/template.py ./mnist_ffn_config_cpu.json normal | tee mnist_ffn_config_cpu_normal.log
python3.6 ./../../benchmark/template.py ./mnist_ffn_config_cpu.json profiled | tee mnist_ffn_config_cpu_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_ffn_config_cpu.json separate | tee mnist_ffn_config_cpu_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_ffn_config_cpu.json | tee mnist_ffn_config_cpu_report.log


python3.6 ./../../benchmark/template.py ./mnist_ffn_config_gpu_1.json normal | tee mnist_ffn_config_gpu_1_normal.log
python3.6 ./../../benchmark/template.py ./mnist_ffn_config_gpu_1.json profiled | tee mnist_ffn_config_gpu_1_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_ffn_config_gpu_1.json separate | tee mnist_ffn_config_gpu_1_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_1.json | tee mnist_ffn_config_gpu_1_report.log

python3.6 ./../../benchmark/template.py ./mnist_ffn_config_gpu_2.json normal | tee mnist_ffn_config_gpu_2_normal.log
python3.6 ./../../benchmark/template.py ./mnist_ffn_config_gpu_2.json profiled | tee mnist_ffn_config_gpu_2_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_ffn_config_gpu_2.json separate | tee mnist_ffn_config_gpu_2_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_ffn_config_gpu_2.json | tee mnist_ffn_config_gpu_2_report.log



echo "##################################################################################################"
echo "###           Running CNN Experiments                                                           ##"
echo "##################################################################################################"

python3.6 ./../../benchmark/template.py ./mnist_cnn_config_cpu.json normal | tee mnist_cnn_config_cpu_normal.log
python3.6 ./../../benchmark/template.py ./mnist_cnn_config_cpu.json profiled | tee mnist_cnn_config_cpu_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_cnn_config_cpu.json separate | tee mnist_cnn_config_cpu_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_cnn_config_cpu.json | tee mnist_cnn_config_cpu_report.log


python3.6 ./../../benchmark/template.py ./mnist_cnn_config_gpu_1.json normal | tee mnist_cnn_config_gpu_1_normal.log
python3.6 ./../../benchmark/template.py ./mnist_cnn_config_gpu_1.json profiled | tee mnist_cnn_config_gpu_1_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_cnn_config_gpu_1.json separate | tee mnist_cnn_config_gpu_1_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_cnn_config_gpu_1.json | tee mnist_cnn_config_gpu_1_report.log

python3.6 ./../../benchmark/template.py ./mnist_cnn_config_gpu_2.json normal | tee mnist_cnn_config_gpu_2_normal.log
python3.6 ./../../benchmark/template.py ./mnist_cnn_config_gpu_2.json profiled | tee mnist_cnn_config_gpu_2_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_cnn_config_gpu_2.json separate | tee mnist_cnn_config_gpu_2_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_cnn_config_gpu_2.json | tee mnist_cnn_config_gpu_2_report.log


echo "##################################################################################################"
echo "###           Running RNN Experiments                                                           ##"
echo "##################################################################################################"
export PYTHONPATH=$PYTHONPATH:/Users/prathyushsp/Git/tensorflow-dataset-api-gpu:/home/prathyush.sp/Git/tensorflow-dataset-api-gpu

python3.6 ./../../benchmark/template.py ./mnist_rnn_config_cpu.json normal | tee mnist_rnn_config_cpu_normal.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_cpu.json profiled | tee mnist_rnn_config_cpu_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_cpu.json separate | tee mnist_rnn_config_cpu_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_rnn_config_cpu.json | tee mnist_rnn_config_cpu_report.log


python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_1.json normal | tee mnist_rnn_config_gpu_1_normal.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_1.json profiled | tee mnist_rnn_config_gpu_1_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_1.json separate | tee mnist_rnn_config_gpu_1_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_rnn_config_gpu_1.json | tee mnist_rnn_config_gpu_1_report.log


python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_2.json normal | tee mnist_rnn_config_gpu_2_normal.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_2.json profiled | tee mnist_rnn_config_gpu_2_profiled.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_2.json separate | tee mnist_rnn_config_gpu_2_separate.log
python3.6 ./../../benchmark/report_generator.py ./mnist_rnn_config_gpu_2.json | tee mnist_rnn_config_gpu_2_report.log