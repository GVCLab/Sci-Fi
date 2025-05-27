# Sci-Fi
Official PyTorch implementation of "Sci-Fi: Symmetric Constraint for Frame Inbetweening"

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Start frame 1</td>
        <td>End frame 2</td>
        <td>Text</td>
        <td>Generated video</td>
    </tr>
  	<tr>
	  <td>
	    <img src=example_input_pairs/input_pair1/start.jpg width="250">
	  </td>
	  <td>
	    <img src=example_input_pairs/input_pair1/end.jpg width="250>
	  </td>
      <td>
	    A group of people dance in the street at night, forming a circle and moving in unison, exuding a sense of community and joy. A woman in an orange jacket is actively engaged in the dance, smiling broadly. The atmosphere is vibrant and festive, with other individuals participating in the dance, contributing to the sense of community and joy. 
	  </td>
	  <td>
     		<image src=cases/gen_5.gif width="250">
	  </td>
  	</tr>
  	<tr>
	  <td>
	    <img src=cases/6.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/66.jpg width="250">
	  </td>
      <td>
	    A man in a white suit stands on a stage, passionately preaching to an audience. The stage is decorated with vases with yellow flowers and a red carpet, creating a formal and engaging atmosphere. The audience is seated and attentive, listening to the speaker. 
	  </td>
	  <td>
	    <image src=cases/gen_6.gif width="250">
	  </td>
  	</tr>
  <tr>
	  <td>
	    <img src=cases/7.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/77.jpg width="250">
	  </td>
      <td>
	    A man in a blue suit is laughing.
	  </td>
	  <td>
	    <img src=cases/gen_7.gif width="250">
	  </td>
  	</tr>
</table >

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
Download the CogVideoX-5B-I2V model (due to fine-tuning, the weights of the transformer denoiser are different from the original) and EF-Net.
The weights are available at [🤗HuggingFace](https://huggingface.co/LiuhanChen/Sci-Fi) and [🤖ModelScope](https://www.modelscope.cn/models/clhxclh/Sci-Fi).

### 3. Launch the inference script!
The example input keyframe pairs are in `examples/` folder, and 
the corresponding generated videos (720x480, 49 frames) are placed in `outputs/` folder.
</br>
To interpolate, run:
```
bash Sci_Fi_frame_inbetweening.sh
```
