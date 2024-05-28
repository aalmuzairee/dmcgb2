

<h1>DMControl Generalization Benchmark 2</span></h1>

This is the official Pytorch implementation of

[A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning](https://aalmuzairee.github.io/SADA/) by

[Abdulaziz Almuzairee](https://aalmuzairee.github.io), [Nicklas Hansen](https://nicklashansen.com), [Henrik I Christensen](https://hichristensen.com) (UC San Diego)</br>


</br><img  width="100%" src="https://github.com/aalmuzairee/SADA/blob/master/static/videos/cinematic.gif"></br>

and the official release of the DMControl Generalization Benchmark 2 (DMC-GB2).

[[Website]](https://aalmuzairee.github.io/SADA/) [[Paper]](https://arxiv.org/abs/2405.17416) 

-----

## Getting Started

### Packages


All package dependencies can be installed with the following commands. We assume that you have access to a GPU with CUDA >=11.0 support:

```
conda env create -f environment.yaml
conda activate sada
```
If building from docker, we recommend using `nvidia/cudagl:11.3.0-base-ubuntu18.04` as the base image.

-----


### Datasets

This repository has dependencies on external datasets. For full functionality, you need to download the following datasets:

- Places365 Dataset: For applying Random Overlay Image Augmentation, we follow [SODA](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) in using the [Places365](http://places2.csail.mit.edu/download.html) dataset 
- DAVIS Dataset: For evaluating on the [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control), the [DAVIS](https://davischallenge.org/davis2017/code.html) dataset is used for video backgrounds

#### Easy Install

We provide utility scripts for installing these datasets in `scripts` folder, which can be run using 

```
scripts/install_places.sh
scripts/install_davis.sh
```

#### Manual Install 

If you prefer manual installation, the Places365 Dataset can be downloaded by running:

```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

The DAVIS dataset can be downloaded by running:

```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```

After downloading and extracting the data, add your dataset directory to the `datasets` list in `cfgs/config.yaml`.

-----

## Example usage

We provide examples on how to train below.

```sh
# Train SADA with all six strong augmentations
python train.py agent=sada task=walker_walk strong_augs=[all]

# Train SVEA with two selected strong augmentations
python train.py agent=svea task=cup_catch strong_augs=[rotate,rotate_shift]

# Train DrQ with no strong augmentations
python train.py agent=drq task=cheetah_run strong_augs=[]
```
 where the log outputs will be: 

 ```sh
 eval    F: 0            S: 0            E: 0            R: 21.10        L: 1,000        T: 0:00:14      FPS: 682.60     M: train  
 train   F: 1,000        S: 500          E: 1            R: 37.50        L: 1,000        T: 0:00:46      FPS: 533.08  
```

with each letter corresponding to:

 ```sh
 F: Frames            S: Env Steps            E: Episode            R: Episode Reward        L: Episode Length        T: Time     FPS: Frames Per Second    M: Mode	I: Intensity  
```


For logging, we recommend configuring [Weights and Biases](https://wandb.ai) (`wandb`) in `cfgs/config.yaml` to track training progress.


-----

## Config options

Please refer to `cfgs/config.yaml` for a full list of options.

#### Algorithms

There are three algorithms that you can choose from:

- `sada` : [SADA (Almuzairee et al., 2024)](https://github.com/aalmuzairee/dmcgb2/)
- `svea` : [SVEA (Hansen et al., 2021)](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)
- `drq` : [DrQ (Kostrikov et al., 2020)](https://github.com/denisyarats/drq)

by setting the `agent` variable in the `cfgs/config.yaml` file.

#### DMC-GB2 Test Distributions

This codebase currently supports **6** continuous control tasks from **DMControl** with 12 test distributions for each task. Supported tasks are: 

| task
| --- 
| `walker_walk`
| `walker_stand`
| `cheetah_run`
| `finger_spin`
| `cartpole_swingup`
| `cup_catch`

which can be set through the `task` variable.

For evaluating generalization throughout training, we provide 12 test distributions for each task:

| Geometric Test Distributions (dmcgb_geo) | Photometric Test Distributions (dmcgb_photo)
| --- | ---
| `rotate_easy` | `color_easy`
| `rotate_hard` | `color_hard`
| `shift_easy` | `video_easy`
| `shift_hard` | `video_hard`
| `rotate_shift_easy` | `color_video_easy`
| `rotate_shift_hard` | `color_video_hard`

which can be set in the `eval_modes` variable in `cfgs/config.yaml`

</br><img  width="100%" src="https://github.com/aalmuzairee/SADA/blob/master/static/images/repo/dmcgb.png"></br>

For final testing after the training is concluded, we provide three options of testing:
- `dmcgb_geo` : for testing on the 6 geometric test distribtuions from DMC-GB
- `dmcgb_photo` : for testing on the 6 photometric test distributions from DMC-GB
- `dcs` : for testing on the Distracting Control Suite 
  
which can be set in the `test_modes` variable in `cfgs/config.yaml`


The `dcs` option refers to a set of challenging test environments from the [Distracting Control Suite](https://arxiv.org/abs/2101.02722) (DCS) that we integrated. We use the implementation of the original [DMC-GB](https://github.com/nicklashansen/dmcontrol-generalization-benchmark/tree/main?tab=readme-ov-file#test-environments) with the alterations they defined.


#### Strong Augmentations

We further provide options to choose the strong augmentation(s) applied during the training in the `strong_augs` list in the `cfgs/config.yaml`. We sample one strong augmentation from the selected set of strong augmentations for each image.

| Geometric Augmentations (geo) | Photometric Augmentations (photo)
| --- | ---
| `rotate` | `conv`
| `shift` | `overlay`
| `rotate_shift` | `conv_overlay`

</br><img  width="100%" src="https://github.com/aalmuzairee/SADA/blob/master/static/images/repo/aug.png"></br>

-----

## Results

<img src="https://github.com/aalmuzairee/SADA/blob/master/static/images/repo/overall_results.png" width=100%/>

-----



## Citation

If you find our work useful, please consider citing our paper:

```
@misc{almuzairee2024recipe,
      title={A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning}, 
      author={Abdulaziz Almuzairee and Nicklas Hansen and Henrik I. Christensen},
      year={2024},
      eprint={2405.17416},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```

If you used DMC-GB2 in your work, please consider citing the [original DMC-GB](https://arxiv.org/abs/2011.13389) as well.

-----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code and datasets, which is subject to their respective licenses.

-----

## Acknowledgements

We'd like to acknowledge the incredible effort and research in the open source community that made this work possible. This codebase was built on the [DrQv2](https://github.com/facebookresearch/drqv2/tree/main) and [DrQ](https://github.com/denisyarats/drq) repos. 
The new test distributions in DMC-GB2 were built on top of the original [DMC-GB](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) implementation. The [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control) has an original implementation, but we use the reformatted implementation by [DMC-GB](https://github.com/nicklashansen/dmcontrol-generalization-benchmark).
The background videos used in the `video_hard` and `color_video_hard` levels are based off a subset of the [RealEstate10K](https://google.github.io/realestate10k/) dataset, which are included in this repository in `envs/dmcgb/data` directory. The logger is based on the [TD-MPC2](https://github.com/nicklashansen/tdmpc2) repo.


