# Evaluation (FID, precision, recall)
Code here is based on [EDM](https://github.com/NVlabs/edm) repo. 

## Setup
- Download the [ImageNet dataset](https://www.image-net.org/) and save it to `DATA_PATH`
- Compute reference statistics (FID) for ImageNet 128x128
```
python dataset_tool.py --source=DATA_PATH --dest=./fid-refs/imagenet-256x256.zip --resolution=256x256 --transform=center-crop --max-images 50000

python fid.py ref --data=./fid-refs/imagenet-256x256.zip --dest=./fid-refs/imagenet-256x256.npz --fid_features=./precision-recall-refs/image_net_256_fid_features_.pt
```
- Compute reference statistics (FID) for ImageNet 256x256
```
python dataset_tool.py --source=DATA_PATH --dest=./fid-refs/imagenet-128x128.zip --resolution=128x128 --transform=center-crop --max-images 50000

python fid.py ref --data=./fid-refs/imagenet-128x128.zip --dest=./fid-refs/imagenet-128x128.npz --fid_features=./precision-recall-refs/image_net_128_fid_features_.pt
```


