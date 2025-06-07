#!/bin/bash

#加载环境，此处加载anaconda环境以及通过anaconda创建的名为myname的环境
# module load anaconda/2020.11
module load cuda/11.3
source activate myname
module load gcc/11.1.0

python start_sampling.py