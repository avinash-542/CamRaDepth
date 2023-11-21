# CamRaDepth: Semantic Guided Depth Estimation Using Monocular Camera and Sparse Radar

## Setting up docker
Note: Using docker is an option if your host machine has ubuntu 20.04 OS. If you have OS other than Ubuntu, it is recommended to use docker. If you have Ubuntu in your machine and like to use that instead of docker, proceed [here](#setting-up-environment)

1. Go through the [Documentation](https://docs.docker.com/desktop/) to install and start the docker engine.

2. Once, docker engine is installed ans started, use the following command to get ubuntu docker image:

```
docker pull ubuntu:<version-number>
```
3. After pulling the ubuntu docker image, make sure the image or container is running. 

4. If the docker container is not running, then use the follwing command to run the container and use it:
```
docker run --gpus all -it --shm-size=16g ubuntu:<tag>
```
5. initially, at first run of docker image, the tag will be the version number used for pulling ubuntu container/image.

6. After running the container, we will notice that the user of the image is named root in the format follows:
```
root@<container-id>:/#
```
7. Here, note the container-id to save all the progress later on.

8. As the container is running and ready to use now, proceed [Here](#setting-up-environment) to work on.
9. After working, to avoid loss of the progress and save the content worked on, exit the docker using:
```
exit
```
10. Now, start the docker container in which the work was done:
```
docker start <container-id>
```
11. Look of the name of the container by identifying it using the container-id in list of containers. To get the list use command:
```
docker ps
```
12. After noting down the name of container worked on, commit the changes or content in the container using:
```
docker commit <image-name> ubuntu:<tag>
```
13. Here, the tage can be anything of user's choice. But remember the tage to use the container with saved progress in future.

14. After commiting changes, stop the docker using:
```
docker stop <image-name>
```
15. If you wish to have the container to saved to the host machine, use the following command:
```
docker save ubuntu:<tag> > <name>.tar
```
16. To load this file(container), use the follwing command:
```
docker load < <name>.tar
```

Note: The container-id, name of container always change each time we run the container. Only the tag won't change and using different tag name while commiting changes actually create another custom container leaving the one you use aside without deleting.

## Setting up environment
The following steps are used to setup favourable environment for the project to work properly.

1. Update the OS/docker image(ubuntu)
```
sudo apt-get update
```
2. Install all necessary packages and tools for ubuntu
```
sudo apt-get install wget
sudo apt-get install git
sudo apt-get install python3.9
sudo apt-get install pip
```
3. Now, clone this repository using following command:
```
git clone <>
```
4. Once cloning is complete, navigate into the cloned repository:
```
cd CamRaDepth
```
5. Setup virtual environemnt and actiavte it:
```
pip install virtualenv

virtualenv -p /usr/bin/python3 caradep

source caradep/bin/activate
```
6. Now, prepare for the data which will be needed by the project.

7. If you are not using docker container, skip to **step 13**. If using docker, as there won't be any GUI navigate with usual flexibility, use the following command:
```
cd ..
git clone https://github.com/DKandrew/NuScenes-Download-CLI.git
```
8. This repository contain commands and code blocks which help us to download the datasets and other necessary for the project.

9. Use the following commands, we can start donwloading the datasets:
```
cd NuScenes-Download-CLI

python download_nuscenes.py --username <username> --password <password>
sh download_commands.txt
```
10. Here, the username and password are the credentials of Nuscenes which you can register [Here](https://www.nuscenes.org/sign-up?prevpath=nuscenes&prevhash=)

11. After, running the above bash command, use the following command to download datasets:
```
python extract_parallel.py
```
12. Downloads might take time as the data needed is huge.

13. Using a ubuntu system gives GUI flexibility. Hence, download data from [Here](https://www.nuscenes.org/nuscenes#download) and extract data.

14. Create a folder next to ```CamRaDepth```, name it "nuscenes_mini" and move the downloaded datasets into this directory. 

## CUDA establishment

1. Use the following commands to setup the CUDA for the project:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb

dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install cuda
```
2. If faced with any error while establishing, refer and go through the steps from [here](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local).

## Torch setup

1. Follow the steps to setup Pytorch which is compatible with the CUDA and python version established.
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

This will install and setup **torch(1.11.0+cu113), torchvision(0.12.0+cu113) and torchaudio(0.11.0)** 



## NuScenes dev-kit setup
1. To process and work with Nuscenes data, we need Nuscenes dev-kit module installed. It can be installed by using
```
cd CamRaDepth
pip install nuscenes-devkit
```
## Packages and modules setup
1. After all the main setup is done, now we need to install remaining packages and modules need to the project. They can be installed by using the following command:
```
pip install -r requirements.txt
```
2. This will ensure all the necessary packages are insatlled.

3. Due to various versions of different package and there nessecity of 0ther packages, **scikit** module may not be detected. If this happens, install the package by using following command:
```
pip install scikit-image
```
4. With this all the setup for proceeding to work with the project are complete.

## External Repos
1. Clone RAFT to ```external/``` and run below script
```
cd external
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT && ./download_models.sh
cd ..
```
2. Clone Panoptic-DeepLab to ```external/```
```
git clone https://github.com/bowenc0221/panoptic-deeplab.git
```

## Implementation
1. Run all the codes and scripts from inside project root directory only i.e., ```CamRaDepth/.```

2. Prepare data using the following command:
```
./scripts/preprocess_data.sh
```
* Adjust ```DATA_DIR``` and ```DATA_VERSION``` in precprocess_data.sh
**Hint:** The external repos are using deprecated functions, and might cause an error. Replacing `torchvision.models.utils` by `from torch.hub import load_state_dict_from_url` can fix them.<br/>
Files:<br/>
 `external/panoptic-deeplab/segmentation/model/backbone/hrnet.py`<br/>
 `external/panoptic-deeplab/segmentation/model/backbone/mnasnet.py`<br/>
 `external/panoptic-deeplab/segmentation/model/backbone/mobilenet.py`<br/>
 `external/panoptic-deeplab/segmentation/model/backbone/resnet.py`<br/>
 `external/panoptic-deeplab/segmentation/model/backbone/xception.py`


3. Use `external/mseg-semantic` from [mseg](https://github.com/mseg-dataset/mseg-semantic) to generate semantic labels for the image:
```bash
mkdir external/mseg && cd external/mseg/
# Download weights
wget --no-check-certificate -O "mseg-3m.pth" "https://github.com/mseg-dataset/mseg-semantic/releases/download/v0.1/mseg-3m-1080p.pth"
# Install mseg-api
git clone https://github.com/mseg-dataset/mseg-api.git
cd mseg-api && sed -i '12s/.*/MSEG_DST_DIR="\/dummy\/path"/' mseg/utils/dataset_config.py
pip install -e .
cd ..
# Install mseg-semantic
git clone https://github.com/mseg-dataset/mseg-semantic.git
cd mseg-semantic && pip install -r requirements.txt
pip install -e .
cd ..
```
  
**Note**: We now assume that the current working directory is `CamRaDepth/externals/mseg-semantic`
  
change line 23 of file 'mseg-semantic/mseg_semantic/utils/img_path_utils.py' to:

```bash
suffix_fpaths = glob.glob(f"{jpg_dir}/*_im.{suffix}")
```

Run inference with the correct data source directory
```bash
CONFIG_PATH="mseg_semantic/config/test/default_config_360_ms.yaml"

python -u mseg_semantic/tool/universal_demo.py \
  --config="$CONFIG_PATH" \
  model_name mseg-3m \
  model_path mseg-3m.pth input_file ../../../nuscenes_mini/prepared_data/
```

Change and combine the labels for the right format
```bash
python scripts/vehicle_seg.py
```

</details>

6. Download pretrained weights:
```bash
mkdir src/checkpoints && cd src/checkpoints
wget  https://syncandshare.lrz.de/dl/fi17pZyWBpZf38uxQ5XcS3/checkpoints.zip
unzip checkpoints.zip -d ..
```
**Note 1**: "FS" - From scratch, "TL" - Transfer learning scheme. <br> 
**Note 2**: It is assumed that the code is run from the main directory, CamRaDepth

</details>

## Run the code by following the belo details and instructions:
### Modes and Expected Outputs:
 Pass the argument `run_mode`:
  * `train`: if `save_model` is set to True, the model will save intermediate checkpoints and Tensorsboard files, to a folder that is specified as `output_dir/arch_name/run_name`. For example: "outputs/CamRaDepth/Base_batchsize_4_run1". Multiple runs could be done to the same folder. The checkpoints are saved according to the best perfromance so far over the validation set, and in thier name specify the val-loss value, for reference.
  Note that each ckpt file would weigh ~350 MB.

  * `test`: Would not save anything to the disk, but only print a summary of the model's performance on the test set, stating measurments such as RMSE for 50m and 100m, MAE, runtime, performance only on edge cases, etc.

<details>
<summary> <h3> Arguments <h3> </summary>

* There are two ways to pass arguments to the model:
  * `conventual`: Use the common method of passing the different arguments through the command line.
  * `manually` (recommended): Under utils/args.py you will find the "Manual settings" section. Uncomment this section, and set your desired values as you wish, much more comfortably.

* Needed arguments for `training`:
  * `save model`: defines if checkpoints and tensorboard files should be saved to disk.
  * `load_ckpt`: A boolean value that defines if a checkpoint should be loaded from disk.
  * `distributed`: A boolean value that defines if the model should be trained in a distributed manner (torch.nn.DataParallel()).
  * `run_mode`: A string our of ["train", "test"], that defines if the model should be trained or tested.
  * `model`: A string out of ["base (rgb)", "base", "supervised_seg", "unsupervised_seg", "sup_unsup_seg", "sup_unsup_seg (rgb)"].
  * `checkpoint`: Should be set if transfer learning is desired. The abs-path to the checkpoint that should be stated.
  * `arch_name`: A string that helps to disntiguish between running modes (e.g. "Debug", "Transformer", etc.)
  * `run_name`: A specific name for a specific run (e.g. "Base_batchsize_6_run_1").
  * `batch_size`: The batch size.
  * `desired_batch_size`: Hardware could be quite limiting, therefore you can set this parameter for "gradients accumelations", meaning that the backprop pipeline will be exceuted every `update_interval` = `desired_batch_size` / `batch_size` iterations.
  * `div_factor`: Is the div_factor argument for the OneCycleOptimzer by PyTorch.
  * `learning_rate`: The learning rate.
  * `cuda_id`: If using cude, and there is more than one GPU in the system. The default --> 0.
  * `num_steps`: Instead of the setting a fixed number of epochs, set the number of running update steps. (Takes the `update_interval` into account).
  * `num_epochs`: If running as specific number of epochs is desired.
  * `rgb_only`: Set the number of input channels to 3.
  * `input_channels`: Number of input channels (default --> 7).
  * `early_stopping_thresh`: The threshold for the early stopping mechanism.

</details>

### Training Example Command
**Note:** Training does only make sence with the full dataset as the mini does not provide enough data for meaningful training.

```bash
python src/main/runner.py --run_mode train --model base --save_model --batch_size 2 --desired_batch_size 6 --num_steps 60000 --run_name 'base_batch(2-6)' --split <created_split_for_the_full_dataset.npy>
```

### Evaluation (inference) Example Command
**With Full nuScenes Dataset:**
```bash
python src/main/runner.py --run_mode test --checkpoint checkpoints/Base_TL.pth --model base
```
**With nuScenes Mini:**
```bash
python src/main/runner.py --run_mode test --checkpoint checkpoints/Base_TL.pth --model base --split <your_mini_split_path> --mini_dataset
```

**Note:** the `mini_dataset` argument is a must for the mini split, as the dataloader will take the entire split is a test set.
The default split file argument (<your_mini_split_path>) is `new_split.npy`

* Needed arguments for `testing` (inference) - look above for a more detailed description:
  * run_mode
  * checkpoint
  * model
  * cuda_id

</details>

## Visualization

Follow the below code for visualization:


```bash
python visualization/visualization.py --vis_num 10
```
This module will create and save to the disk a variety of inputs, such as the depth map, RADAR and LiDAR projection onto the image place, transperent depth projection, semantic segmenation, etc for each one of the input instances of the given dataset (for `vis_num` different inputs). One could easily view the different visualizations under `output_dir/visualizations/collage` as a single collage, or go to the corresponding directory with the specific instance name, that one would like to examine.


