## Cài đặt

Yêu cầu: Python > 3.7

#### Cài đặt các thư viện sử dụng:

```bash
conda create -n training python=3.7
conda activate training
pip install -r requirement.txt
```

## Docker (dev)
#### Build docker image
```
docker build -t opencv_training .
```
#### Start new docker container
```
docker run -it -d --restart always --shm-size=32GB --gpus=all -v `pwd`/out_snapshot:/workspace/out_snapshot -v `pwd`/dataset:/workspace/dataset --name opencv_training opencv_training:latest
```
#### Run training ([resnet18, resnet34, resnet50, mobilenet_v2, mobilenet_v3_small])
```
 python train.py --network resnet18 --num_classes 2 --input_size 224 --epochs 20 --learning_rate 0.0005 --batch_size 64 --dataset dataset/mvtec --path_pretrain pretrains/resnet18-5c106cde.pth --job_name obj_classify
```

#### Run export model
```
 python export_onnx.py --network resnet18 --num_classes 2 --input_size 224 --path_model out_snapshot/resnet18/resnet18_best_model.pth
```
