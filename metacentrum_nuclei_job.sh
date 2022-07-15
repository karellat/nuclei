#!/usr/bin/sh
#PBS -l select=1:ncpus=1:mem=10gb:ngpus=1:gpu_mem=30gb:scratch_local=30gb
#PBS -l walltime=6:00:00
#PBS -q gpu
#PBS -m ae

# FUNCS
find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

# MODULES
echo "Loading modules"
module load conda-modules octave

# CONDA
if find_in_conda_env ".*nuclei.*"; then
   echo "Environment nuclei exists" 
else
   echo "Creating environment nuclei"
   conda env create -f environment.yml
fi

echo "Preparing storage" 

# STORAGE
DATADIR="/storage/brno3-cerit/home/karellat/nuclei"
RESULTDIR="$DATADIR/$PBS_JOBID"
NUCLEIDIR="$SCRATCHDIR/nuclei"

# clean the SCRATCH when job finishes (and data
# are successfully copied out) or is killed
trap 'clean_scratch' TERM EXIT

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }
cp -r "$DATADIR"  "$SCRATCHDIR" || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cd "$NUCLEIDIR"
ls -l .


# Run script
echo "Running training script"
conda run --no-capture-output -n nuclei python script_training.py
echo "Running worm script"
conda run --no-capture-output -n nuclei python script_worm.py

echo "Copying result files"
mkdir "$RESULTDIR"
mv "$NUCLEIDIR/results" "$RESULTDIR" 
# Copying results
clean_scratch
