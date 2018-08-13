#!/bin/bash

#SBATCH --partition=general
#SBATCH --job-name=javid
#SBATCH --ntasks=1 --nodes=4
#SBATCH --mem-per-cpu=55000 
#SBATCH --mem-=550000
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email
python train.py -file rest1_LR.mat.csv  unlabled.csv -m svr -semi svr #isotonic #svr
#python train.py -file rest1_LR.mat.csv -m svr
