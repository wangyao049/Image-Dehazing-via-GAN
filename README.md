# Image-Dehazing-via-GAN
This is the implement of the master graduation thesis.

## Requirement:
1. python 3.5
2. TensorFlow 1.0.0
3. OpenCV

## Get Starting:

1. Estimating the Depth map of the hazy image

'''
python gan.py \
--input_dir test_img
--output_dir test
--mode test \
--checkpoint dehaze_train
'''

2. Single image dehazing

'''
python dehaze.py \
--img_dir *** \
--depth_dir *** \
--beta *
'''
