# Synthetic Dust Generator

The code generates artificial dust-like particles that are heterogeneously distributed. Additionally, our method simulates the temporal flow of dust generated during tillage. Starting with a dust-free image, progressing to a dense dust appearance.

Further information is available at: https://agriscapes-dataset.com/


![Synthetic_dust_generation_method](img_github/dust_method_overview.png)

## Installation

Please first install the requirements mentioned below:

```bash
    python setup.py install
    pip install timm
    pip install natsort
```
    
## Documentation

Available masks and their path. Please replace "path_to_mask_image.png" with the corresponding masks path.
```javascript
1. Tailwind blowing the dust towards the tractor covering the entire image.       path: masks/mask_all.png
2. Wind blows dust to the left.  -> Right side of the image dust-free.            path: masks/mask_left.png
3. Wind blows dust to the right. -> Left side of the image dust-free.             path: masks/mask_right.png
4. No wind or headwind. -> Only in the middle of the image is dust generated      path: masks/mask_middle.png
```

Further information can be found in our publication.
## Usage
General command to generate synthetic dusty images: 
```javascript
python depth_est.py --image_input_path input/ --image_output_path output/ --wind_mask_path path_to_mask_image.png
```
Please replace "path_to_mask_image.png" for the corresponding mask you wanna use.



## Demo

Input image sequence:
![Synthetic_dust_generation_method](img_github/input.gif)
Output generated with the "mask_middle.png" mask:
![Synthetic_dust_generation_method](img_github/output_middle.gif)


## Acknowledgements

 - [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/) was used in this work for single image depth prediciton.
 - The [Perlin noise implementation](https://github.com/caseman/noise) of Casey Duncan was used in this work.


## Citation

If you find this work useful for your research, please cite our paper: 

```bibtex
OUR Paper
```


Please also cite the work of the developers whose code was used in this method.

Single image depth prediction method:
```bibtex
@ARTICLE{9178977,
  author={Ranftl, René and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer}, 
  year={2022},
  volume={44},
  number={3},
  pages={1623-1637},
  keywords={Training;Estimation;Three-dimensional displays;Cameras;Videos;Measurement;Motion pictures;Monocular depth estimation;single-image depth prediction;zero-shot cross-dataset transfer;multi-dataset training},
  doi={10.1109/TPAMI.2020.3019967}
}
@INPROCEEDINGS{9711226,
  author={Ranftl, René and Bochkovskiy, Alexey and Koltun, Vladlen},
  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={Vision Transformers for Dense Prediction}, 
  year={2021},
  volume={},
  number={},
  pages={12159-12168},
  keywords={Computer vision;Image resolution;Semantics;Neural networks;Estimation;Training data;Computer architecture;Machine learning architectures and formulations;3D from a single image and shape-from-x;Segmentation;grouping and shape},
  doi={10.1109/ICCV48922.2021.01196}
}
```
Please also consider to cite the Perlin noise paper:
```bibtex
@ARTICLE{10.1145/566654.566636,
author = {Perlin, Ken},
title = {Improving noise},
year = {2002},
issue_date = {July 2002},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {21},
number = {3},
issn = {0730-0301},
url = {https://doi.org/10.1145/566654.566636},
doi = {10.1145/566654.566636},
abstract = {Two deficiencies in the original Noise algorithm are corrected: second order interpolation discontinuity and unoptimal gradient computation. With these defects corrected, Noise both looks better and runs faster. The latter change also makes it easier to define a uniform mathematical reference standard.},
journal = {ACM Trans. Graph.},
month = jul,
pages = {681–682},
numpages = {2},
keywords = {procedural texture}
}
```
## License

[MIT](https://choosealicense.com/licenses/mit/)

