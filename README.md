# Spatial-Temporal Super-Resolution of Satellite Imagery via Conditional Pixel Synthesis
PyTorch implementation of NeurIPS 2021 paper "Spatial-Temporal Super-Resolution of Satellite Imagery via Conditional Pixel Synthesis"

**Note: Still under construction:)!**

## Requirements

pip install -r requirements.txt

## Usage


To train EAD on Texas housing dataset please run:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train_3dis.py --path TRAIN_PATH --test_path TEST_PATH --output_dir OUTPUT_DIR
```
To train EA64 on Texas housing dataset please run:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train_3dis_attpatch.py --path TRAIN_PATH --test_path TEST_PATH --output_dir OUTPUT_DIR
```
To train EAD on FMoW-Sentinel2 crop field dataset please run:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 fmow_train_3dis.py --path TRAIN_PATH --test_path TEST_PATH --output_dir OUTPUT_DIR
```
To train EA64 on FMoW-Sentinel2 crop field dataset please run:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 fmow_train_3dis_attpatch.py --path TRAIN_PATH --test_path TEST_PATH --output_dir OUTPUT_DIR
```

## Datasets
[Texas Housing Dataset](https://drive.google.com/drive/folders/1rFjxWxpH_4SCa30y58e3OJnH0uIyiMuD?usp=sharing)   
[FMoW-Sentinel Crop Field Dataset](https://drive.google.com/drive/folders/1DLDU4vVU37xZNy-a10yum8oZhVKBoQLk?usp=sharing)

## Citation

If you find our work or datasets useful, please cite the following paper:
```
@inproceedings{he2021spatial,
  title={Spatial-Temporal Super-Resolution of Satellite Imagery via Conditional Pixel Synthesis},
  author={He, Yutong and Wang, Dingjie and Lai, Nicholas and Zhang, William and Meng, Chenlin and Burke, Marshall and Lobell, David B. and Ermon, Stefano},
  year={2021},
  month={December},
  abbr={NeurIPS 2021},
  booktitle={Neural Information Processing Systems},
}

```

The code is based on the [styleganv2 pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) and [CIPS pytorch implementation](https://github.com/saic-mdal/CIPS).

Nvidia-licensed CUDA kernels (fused_bias_act_kernel.cu, upfirdn2d_kernel.cu) are for non-commercial use only.

