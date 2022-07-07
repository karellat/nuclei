#!/usr/bin/sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=4:00:00
#PBS -q gpu
#PBS -N nuclei

# STORAGE
DATADIR="/storage/brno3-cerit/home/karellat/nuclei"

# clean the SCRATCH when job finishes (and data
# are successfully copied out) or is killed
trap 'clean_scratch' TERM EXIT

cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR
ls $SCRATCHDIR

# DEPENDENCIES
module load conda-modules
module load octave
conda env create -f environment.yml
conda run --no-capture-output -n nuclei python script_worm.py

# Run script
# Arguments