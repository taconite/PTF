# Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration 
This repository contains the implementation of our paper
[Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration ](https://arxiv.org/abs/2104.08160). The code is largely based on [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://github.com/autonomousvision/occupancy_networks).

You can find detailed usage instructions for training your own models and using pretrained models below.

If you find our code useful, please consider citing:

```bibtex
@InProceedings{PTF:CVPR:2021,
    author = {Shaofei Wang and Andreas Geiger and Siyu Tang},
    title = {Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration},
    booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```

## Installation
This repository has been tested on the following platforms:
1) Python 3.7, PyTorch 1.6 with CUDA 10.2 and cuDNN 7.6.5, Ubuntu 20.04
2) Python 3.7, PyTorch 1.6 with CUDA 10.1 and cuDNN 7.6.4, CentOS 7.9.2009

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `PTF` using
```
conda env create -n PTF python=3.7
conda activate PTF
```

Second, install PyTorch 1.6 via the [official PyTorch website](https://pytorch.org/get-started/previous-versions/).

Third, install dependencies via
```
pip install -r requirements.txt
```

Fourth, manually install [pytorch-scatter](https://github.com/rusty1s/pytorch_scatter).

Lastly, compile the extension modules. You can do this via
```
python setup.py build_ext --inplace
```

(Optional) if you want to use the registration code under `smpl_registration/`, you need to install kaolin. Download the code from the [kaolin repository](https://github.com/NVIDIAGameWorks/kaolin.git), checkout to commit e7e513173bd4159ae45be6b3e156a3ad156a3eb9 and install it according to the instructions.

(Optional) if you want to train/evaluate single-view models (which corresponds to configurations in `configs/cape_sv`), you need to install OpenDR to render depth images. You need to first install OSMesa, here is the command of installing it on Ubuntu:
```
sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev libosmesa6-dev
```
For installing OSMesa on CentOS 7, please check [this related issue](https://github.com/openai/mujoco-py/issues/96#issuecomment-493391273). After installing OSMesa, install OpenDR via:
```
pip install opendr
```

## Build the dataset
To prepare the dataset for training/evaluation, you have to first download the CAPE dataset from the [CAPE website](https://cape.is.tue.mpg.de/dataset).

0. Download [SMPL v1.0](https://smpl.is.tue.mpg.de/), clean-up the chumpy objects inside the models using [this code](https://github.com/vchoutas/smplx/tree/master/tools), and rename the files and extract them to `./body_models/smpl/`, eventually, the `./body_models` folder should have the following structure:
   ```
   body_models
    └-- smpl
		├-- male
		|   └-- model.pkl
		└-- female
		    └-- model.pkl

   ```
Besides the SMPL models, you will also need to download all the .pkl files from [IP-Net repository](https://github.com/bharat-b7/IPNet/tree/master/assets) and put them under `./body_models/misc/`. Finally, run the following script to extract necessary SMPL parameters used in our code:
```
python extract_smpl_parameters.py
```
The extracted SMPL parameters will be save into `./body_models/misc/`.
1. Extract CAPE dataset to an arbitrary path, denoted as ${CAPE_ROOT}. The extracted dataset should have the following structure:
   ```
   ${CAPE_ROOT}
    ├-- 00032
	├-- 00096
	|   ...
	├-- 03394
	└-- cape_release

   ```
2. Create `data` directory under the project directory.
3. Modify the parameters in `preprocess/build_dataset.sh` accordingly (i.e. modify the --dataset_path to ${CAPE_ROOT}) to extract training/evaluation data.
4. Run `preprocess/build_dataset.sh` to preprocess the CAPE dataset.

## Pre-trained models
We provide pre-trained [PTF and IP-Net models](https://drive.google.com/drive/folders/1ZKaPXvjDyp4sLwgkM6cWXOny01glh88N?usp=sharing) with two encoder resolutions, that is, 64x3 and 128x3. After downloading them, please put them under respective directories `./out/cape` or `./out/cape_sv`.

## Generating Meshes
To generate all evaluation meshes using a trained model, use
```
python generate.py configs/cape/{config}.yaml
```
Alternatively, if you want to parallelize the generation on a HPC cluster, use:
```
python generate.py --subject-idx ${SUBJECT_IDX} --sequence-idx ${SEQUENCE_IDX} configs/cape/${config}.yaml
```
to generate meshes for specified subject/sequence combination. A list of all subject/sequence combinations can be found in `./misc/subject_sequence.txt`. 

## SMPL/SMPL+D Registration
To register SMPL/SMPL+D models to the generated meshes, use either of the following:
```
python smpl_registration/fit_SMPLD_PTFs.py --num-joints 24 --use-parts --init-pose configs/cape/${config}.yaml # for PTF
python smpl_registration/fit_SMPLD_PTFs.py --num-joints 14 --use-parts configs/cape/${config}.yaml # for IP-Net
```
Note that registration is very slow, taking roughly 1-2 minutes per frame. If you have access to HPC cluster, it is advised to parallelize over subject/sequence combinations using the same subject/sequence input arguments for generating meshes.

## Training
Finally, to train a new network from scratch, run
```
python train.py --num_workers 8 configs/cape/${config}.yaml
```

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
tensorboard --logdir ${OUTPUT_DIR}/logs --port 6006
```
where you replace ${OUTPUT_DIR} with the respective output directory.

## License
We employ [MIT License](LICENSE.md) for the PTF code, which covers
```
extract_smpl_parameters.py
generate.py
train.py
setup.py
im2mesh/
preprocess/
```

Modules not covered by our license are modified versions from [IP-Net](https://github.com/bharat-b7/IPNet) (`./smpl_registration`) and [SMPL-X](https://github.com/nghorbani/human_body_prior) (`./human_body_prior`); for these parts, please consult their respective licenses and cite the respective papers.
