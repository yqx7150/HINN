# HINN
The Code is created based on the method described in the following paper:
Neural holography based on invertible networks  
Author: XU Xiao-ling, LI He-chen, PENG Hong, WAN Wen-bo, WANG Yu-hao, LIU Qie-gen  
Date : Oct. 27, 2021  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2021, Department of Electronic Information Engineering, Nanchang University.


## HINN system overview and network architecture
 <div align="center"><img src="https://github.com/yqx7150/HINN/blob/main/figs/1.png"> </div>
 
## Installation 
We have tested our code on Ubuntu 18.04 LTS with PyTorch 1.4.0, CUDA 10.1 and cudnn7.6.5. Please install dependencies by 
conda env create -f environment.yml
 
# Train
Prepare your own datasets for HINN

You need to create a folder for the amplitude data set, and the corresponding phase data set, and then change the path to import the data set.

##command
sh train.sh

# Test
Put  data set that is not a training set

##command
sh test.sh

# Acknowledgement
The code is based on [yzxing87/Invertible-ISP](https://github.com/yzxing87/Invertible-ISP)


