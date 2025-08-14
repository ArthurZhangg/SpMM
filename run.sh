#!/bin/bash

# 定义参数列表，格式为数组，里面每个元素是4个参数组成的字符串
param_list=(
    "197 2304 768 0.9"
    "197 768 3072 0.9"
    "197 3072 768 0.9"
    "197 768 768 0.9"
)

for params in "${param_list[@]}"
do
    echo "Running command with parameters: $params"

    # 对同一组参数，执行5次命令
    for i in {1..5}
    do
        echo "  Run #$i"
        ./test $params
    done
done

echo "All runs completed."
