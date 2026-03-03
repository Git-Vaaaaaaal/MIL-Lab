#!/bin/ksh 
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N clamv2
cd $WORKDIR
cd /beegfs/data/work/c-2iia/vb710264/mil_lab
source /beegfs/data/work/c-2iia/vb710264/mil_lab/milab_venv/bin/activate
module load python
export PYTHONPATH=/work/c-2iia/vb710264/mil_lab/milab_venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib

python /beegfs/data/work/c-2iia/vb710264/mil_lab/MIL-Lab/train.py