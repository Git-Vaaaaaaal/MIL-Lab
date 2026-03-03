#!/bin/ksh -1
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N mil_train
cd $WORKDIR
cd /beegfs/data/work/c-2iia/vb710264/mil_lab

module load python

source /beegfs/data/work/c-2iia/vb710264/mil_lab/milab_venv/bin/activate

export PYTHONPATH=/work/c-2iia/vb710264/mil_lab/milab_venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib
pip3 list
cd cd /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab
python /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab/src/train.py