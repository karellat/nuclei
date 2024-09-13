# 3D Non-separable Moment Invariants and Their Use in Neural Networks
## Abstract
Recognition of 3D objects is an important task in many bio-medical and industrial applications. The recognition algorithms should work regardless of a particular orientation of the object in the space.
In this paper,
we introduce new 3D rotation moment invariants, which are composed of non-separable Appell moments.
We show  that non-separable moments may outperform the separable ones in terms of recognition power and robustness thanks to a~better distribution of their zero surfaces over the image space.
We test the numerical properties and discrimination power of the proposed invariants on three real datasets -- MRI images of human brain, 3D scans of statues, and confocal microscope images of worms.
We show the robustness to resampling errors improved more than twice and the recognition rate increased by 2 - 10 \% comparing to most common descriptors.
In the last section, we show how these invariants can be used in state-of-the-art neural networks for image recognition. The proposed  H-NeXtA  architecture improved the recognition rate by 2 - 5 \% over the current networks.

This repository accompanies the paper "3D Non-separable Moment Invariants and Their Use in Neural Networks" and includes the code for the nuclei classification experiments. 
For the neural network experiments, we use the [H-NeXt](https://arxiv.org/abs/2311.01111) architecture. The code for H-NeXt is available in the [H-NeXt repository](https://github.com/karellat/h-next). 

## Preliminaries
The code is written in Python 3.10. To install the required package use [conda](https://docs.conda.io/en/latest/).

```bash
# Create the python environment
conda env create -f environment.yml
conda activate nuclei 
# Download the data
chmod +x ./data/get_data.sh
./data/get_data.sh
```
## Code structure
* [./matlablib](./matlablib) - Matlab code for the moment invariants calculation
* [./appell_invariants.py](./appell_invariants.py) - Python code for the Appell invariants calculation
* [./appell_polynomials_3D.py](./appell_polynomials_3D.py) - Python code for the Appell polynomials calculation
* [./invariant3d.py](./invariant3d.py) - Python code for the other widely-used invariants (Moment, Zernike, Gaus-Hermite, ...) calculation
* [./data](./data) - Data for the nuclei classification experiments
* config_* - Configuration files for the nuclei classification experiments

## Running the experiment
To run the nuclei classification experiment with Appell Invariants, use the following commands.

### To train the classifier 
```bash
python script_training.py
```
### To test the classifier
```bash
python script_worm.py
```
The results are stored in the 'results_{NAME}.mat' file, where the dictionary contains the following keys:
* 'minimal_distances' - minimal distances between the test and training samples
* 'argmin_distances' - index of the training samples with the minimal distance


