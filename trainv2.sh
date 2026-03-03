#!/bin/ksh
#$ -q gpu
#$ -j y
#$ -o resultv2.out
#$ -N mil_trainv2
cd /beegfs/data/work/c-2iia/vb710264/mil_lab
module load python
source /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab/.venv/bin/activate
export PYTHONPATH=/work/c-2iia/vb710264/mil_lab/MIL-Lab/.venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib
cd /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab
python /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab/src/train.py