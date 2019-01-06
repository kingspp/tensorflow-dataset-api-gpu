#!/usr/bin/env bash

echo "##################################################################################################"
echo "###           Running RNN Experiments                                                           ##"
echo "##################################################################################################"
export PYTHONPATH=$PYTHONPATH:/Users/prathyushsp/Git/tensorflow-dataset-api-gpu:/home/prathyush.sp/Git/tensorflow-dataset-api-gpu

python3.6 ./../../benchmark/template.py ./mnist_rnn_config_cpu.json normal | tee mnist_rnn_config_cpu.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_1.json normal | tee mnist_rnn_config_gpu_1.log
python3.6 ./../../benchmark/template.py ./mnist_rnn_config_gpu_2.json normal | tee mnist_rnn_config_gpu_2.log
python3.6 ./../../benchmark/report_generator.py ./mnist_rnn_config_cpu.json | tee report.log

#
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_1.json normal
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_1.json profiled
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_1.json separate
#python3.6 ./../../../benchmark/report_generator.py ./mnist_rnn_config_gpu_1.json
#
#
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_2.json normal
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_2.json profiled
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_2.json separate
#python3.6 ./../../../benchmark/report_generator.py ./mnist_rnn_config_gpu_2.json


#
#echo "##################################################################################################"
#echo "###           Running FFN Experiments                                                           ##"
#echo "##################################################################################################"
#export PYTHONPATH=/Users/prathyushsp/Git/tensorflow-dataset-api-gpu:/home/prathyush.sp/Git/tensorflow-dataset-api-gpu
#
#python3.6 ./../../../benchmark/template.py ./mnist_ffn_config_cpu.json normal
#python3.6 ./../../../benchmark/template.py ./mnist_ffn_config_gpu_1.json profiled
#python3.6 ./../../../benchmark/template.py ./mniff separate
#python3.6 ./../../../benchmark/report_generator.py ./mnist_rnn_config_cpu.json
#
#
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_1.json normal
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_1.json profiled
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_1.json separate
#python3.6 ./../../../benchmark/report_generator.py ./mnist_rnn_config_gpu_1.json


#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_2.json normal
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_2.json profiled
#python3.6 ./../../../benchmark/template.py ./mnist_rnn_config_gpu_2.json separate
#python3.6 ./../../../benchmark/report_generator.py ./mnist_rnn_config_gpu_2.json