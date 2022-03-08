RayMVSNet
===
RayMVSNet: Learning Ray-based 1D Implicit Fields for Accurate Multi-View Stereo
---

# How to use

## Environment
* CUDA 11.2
* Python 3.8.5
* torch 1.7.1+cu110
* pip install -r requirements.txt

## Data
* Download the preprocessed [DTU](https://drive.google.com/file/d/1Mfx1oDoAzPbiqfseD8r02czPaNjUoUMJ/view) data and unzip it to data/dtu.
``` 
./dtu  
      ├── Eval                 
      │
      ├── Rectified                 
      │   ├── scan1_train       
      │   ├── scan2_train       
      │   └── ...                
      ├── Cameras
      │   ├── pair.txt   
      │   ├── train   
      │       ├── 00000000_cam.txt   
      │       ├── 00000001_cam.txt   
      │       └── ...  
      └── Depths4         
          ├── scan1_train   
          ├── scan2_train    
          └── ... 
```     
## Training
* python train.py

## Testing
* python test.py

## Evaluation
