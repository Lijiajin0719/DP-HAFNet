# DP-HAFNet

DP-HAFNet is a novel deep learning framework for high-quality ultrasound image reconstruction from single plane-wave radio frequency (RF) data. This network features a dual-path architecture that simultaneously predicts pixel-level adaptive weights and channel-weighted beamforming outputs, enabling superior image quality while maintaining computational efficiency.

## 📄 Corresponding Publication
This repository contains the implementation of the method described in our paper:

***DP-HAFNet: A dual-path hierarchical adaptive fusion network for ultrasound image reconstruction from single plane-wave RF data***

## 🛠 Environment Setup

This project requires Python 3.8 and PyTorch 2.0.1. Follow these steps to set up the environment:

### 1. Clone Repository
```bash
git clone https://github.com/Lijiajin0719/DP-HAFNet.git
cd DP-HAFNet
```
### 2. Create Conda Environment
```bash
conda create -n dp-hafnet python=3.8
conda activate dp-hafnet
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
## 📊 Dataset
1.Download datasets from **[CUBDL](https://ieee-dataport.org/competitions/challenge-ultrasound-beamforming-deep-learning-cubdl-datasets)** and **[PICMUS](https://www.ustb.no/ustb-datasets/)**

2.Extract files to the **data** directory

*If you use this dataset, please cite the following article:*

**CUBDL**
```bash
@INPROCEEDINGS{9251434,
  author={Bell, Muyinatu A. Lediju and Huang, Jiaqi and Hyun, Dongwoon and Eldar, Yonina C. and van Sloun, Ruud and Mischi, Massimo},
  booktitle={2020 IEEE International Ultrasonics Symposium (IUS)}, 
  title={Challenge on Ultrasound Beamforming with Deep Learning (CUBDL)}, 
  year={2020},
  pages={1-5},
  keywords={Deep learning;Measurement;Image quality;Ultrasonic imaging;Array signal processing;Benchmark testing;Task analysis},
  doi={10.1109/IUS46767.2020.9251434}
}
```
**PICMUS**
```bash
@INPROCEEDINGS{7728908,
  author={Liebgott, H. and Rodriguez-Molares, A. and Cervenansky, F. and Jensen, J.A. and Bernard, O.},
  booktitle={2016 IEEE International Ultrasonics Symposium (IUS)}, 
  title={Plane-Wave Imaging Challenge in Medical Ultrasound}, 
  year={2016},
  pages={1-4},
  keywords={Ultrasonic imaging;Measurement;Biomedical imaging;Image resolution;Speckle;Array signal processing;ultrasound;challenge;Plane-Wave;beamforming;ultrafast},
  doi={10.1109/ULTSYM.2016.7728908}
}
```

## 🏋️ DP-HAFNet train/test
### 1. Train
```bash
python train.py --num_epochs 2000 --batch_size 8 --gpu_id 0 --log_dir logs
```
To see more intermediate results, check out `./logs/DP_HAFNet...`.
### 2. Test
```bash
python test.py --model_path logs/... --gpu_id 0 --log_dir logs --save_images --save_mat
```
The test results will be saved to file here: `./logs/test...`.

## 📧 Contact
For any questions regarding the paper or this implementation, please feel free to contact the authors.

📩 **Email:** [15563866837@163.com](15563866837@163.com)

---
*🌟 We appreciate your interest in our work!*
