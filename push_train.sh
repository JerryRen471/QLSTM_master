#!/bin/bash

#加载环境，此处加载anaconda环境以及通过anaconda创建的名为myname的环境
# module load anaconda/2020.11
module load cuda/11.3
source activate myname
module load gcc/11.1.0

# Ns_list=(10000)
# Nm_list=(10 20 30 40 50 60 70 80 90 100)
Ns_list=(10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
Nm_list=(100)

for i in {1..5}; do
    for Ns in ${Ns_list[@]}; do
        for Nm in ${Nm_list[@]}; do
            python SimpLPS_rand30new.py --Ns $Ns --Nm $Nm
        done
    done
done