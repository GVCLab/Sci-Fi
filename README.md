# Sci-Fi
Official PyTorch implementation of "Sci-Fi: Symmetric Constraint for Frame Inbetweening"

## Deployment for frame inbetweening
### 1. Setup repository and environment
```
git clone https://github.com/GVCLab/Sci-Fi.git
cd Sci-Fi
conda create -n Sci-Fi python==3.12
conda activate Sci-Fi
pip install -r requirements.txt
```
### 2. Download checkpoint
Download the CogVideoX-I2V-5B model (due to fine-tuning, the weights of the transformer denoiser are different from the original) and EF-Net. [checkpoint](https://drive.google.com/drive/folders/1H7vgiNVbxSeeleyJOqhoyRbJ97kGWGOK?usp=sharing)

### 3. Launch the inference script!
The example input keyframe pairs are in `examples/` folder, and 
the corresponding generated videos (720x480, 49 frames) are placed in `outputs/` folder.
</br>
To interpolate, run:
```
bash Sci_Fi_frame_inbetweening.sh
```
