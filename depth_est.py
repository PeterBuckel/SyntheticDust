import os
import argparse
import numpy as np
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import math
import natsort 
import random
from noise import pnoise3

def parse_args():
    parser = argparse.ArgumentParser(
        description='Function for dust/haze synthesis.')

    parser.add_argument('--image_input_path', type=str,
                        help='path to a input image folder', required=True)
    parser.add_argument('--image_output_path', type=str,
                        help='path to a output image folder', required=True)
    parser.add_argument('--wind_mask_path', type=str,
                        help='path to a mask image for wind', required=True)
    return parser.parse_args()

def gen_dust(clean_img, depth_img, perlin_noise, mask, beta, A):
    mask = mask/255
    
    ### depth image normalization and apply mask
    #depth_img = 255 - depth_img
    depth_img = depth_img/255
    #depth_img = normalize(depth_img)
    
    ### Apply perlin noise
    dst_3c = np.zeros([1860,2880,3], dtype=np.float64)
    
    ### combine depth image and perlin noise
    dst = depth_img + perlin_noise
    cv2.imwrite("output/test_dst.png", dst*255)
    #dst = perlin_noise
    dst = dst * mask
    cv2.imwrite("output/test_mask.png", dst*255)
    dst_3c[:,:,0] = dst
    dst_3c[:,:,1] = dst
    dst_3c[:,:,2] = dst
    trans = np.exp(-beta*dst_3c)
    hazy = clean_img*trans + A*(1-trans)
    hazy = np.array(hazy, dtype=np.uint8)
    return hazy

def depth_est(img):
    ### Depth estimation based on https://pytorch.org/hub/intelisl_midas_v2/
    model_type = "DPT_Large"

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    print('-> depth estimation Done!')
    return output
    
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    #0.1% of pixels
    numpx = int(max(math.floor(imsz/1000),1))
    #flatten images
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)
    #sort and get values of brightest pixels
    indices = darkvec.argsort()[-numpx:]
    A_ind = imvec[indices]
    #mean value of brightest pixels
    A = A_ind.mean(0)
    return A

def normalize(img):
    max_value = img.max()
    min_value = img.min()
    img_new = (img - min_value) / (max_value - min_value)
    return img_new

def DCP(img,sz):
    d_min = ndimage.minimum_filter(img, footprint=np.ones((sz, sz,3)), mode='nearest')
    d = d_min.min(axis=2)
    return d

def sigmoid(x,c,a):
    return 3/(1 + math.exp(-a*(x-c)))

if __name__ == '__main__':
    args = parse_args()
    ### load input images and sort 
    img_fold = os.listdir(args.image_input_path)
    input_img_path = natsort.natsorted(img_fold,reverse=False)

    ### Dimension for improved Perlin Noise
    width, height = 2880, 1860
    frames = len(input_img_path)
    # Create an empty array to store the noise values
    noise_map = np.zeros((height, width, frames))
    frame_sub = int(len(input_img_path)/2)
    ### Generate Improved Perlin Noise
    for i in range(height):
        for j in range(width):
            for k in range(frames):
                noise_map[i][j][k] = pnoise3(i / 2000, j / 2000, k / frame_sub, octaves=8, persistence=0.5, lacunarity=2.0, repeatx=height, repeaty=width, repeatz=frames, base=42)
      
    ### for beta sigmoid 
    z = np.linspace(0, 40, len(input_img_path)) #(min,max,n) n: number of images per sequence
    count = 1
    for i in range(len(input_img_path)):
        print(i)
        ### Read input images
        input_img = cv2.imread(os.path.join(args.image_input_path, input_img_path[i]), cv2.IMREAD_COLOR)
        a = os.path.join(args.image_input_path, input_img_path[i])
        print( input_img_path[i])
        ### Global atmospheric light based on the approximation of 100 dusty images
        atm = [211, 208, 203]
               
        ### Load Perlin Noise and Normalize 0 to 1
        noise_i_2d = noise_map[:,:,i]
        perlin_noise = normalize(noise_i_2d)
     
        ### Generate depth images
        # Downscale to half to speed up
        input_img_down = cv2.resize(input_img, (1440, 960)) 
        # Depth estimation
        depth_img = depth_est(input_img_down)
        # Upscale
        depth_img = cv2.resize(depth_img, (2880, 1860)) 
        cv2.imwrite("output/test_depth.png", depth_img)
        cv2.imwrite("output/test_perlin.png", perlin_noise*255)
        ### Randomize values
        beta_sigmoid = sigmoid(z[i],10,0.5)      

        mask_middle = cv2.imread(args.wind_mask_path, cv2.IMREAD_GRAYSCALE)
        dusty_middle = gen_dust(input_img, depth_img, perlin_noise, mask_middle, beta=beta_sigmoid, A=atm)
        name=args.image_output_path + "%04d.png" % count
        cv2.imwrite(name, dusty_middle)
        
        count = count + 1
        
        

