RayMVSNet
===
RayMVSNet: Learning Ray-based 1D Implicit Fields for Accurate Multi-View Stereo

Junhua Xi* Yifei Shi* Yijie Wang Yulan Guo Kai Xu†

National University of Defense Technology

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
      └── Depths         
          ├── scan1_train   
          ├── scan2_train    
          └── ... 
```     
## Training
* Train the network
* python train.py

## Testing
* python test.py

You can test with the pretrained model:  ./model.ckpt

## Depth Fusion
* python fusion.py

## Evaluation
* Download the offical evaluation tool from [DTU benchmark](http://roboimagedata.compute.dtu.dk/?page_id=36)
* We provide our pre-computed point clouds for your convenience
