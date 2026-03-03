#!/bin/ksh -1
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N mil_train
cd $WORKDIR
cd /beegfs/data/work/c-2iia/vb710264/mil_lab
source /beegfs/data/work/c-2iia/vb710264/mil_lab/milab_venv/bin/activate

export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib

python /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab/src/train.py