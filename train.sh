docker run -v `pwd`/out_snapshot:/workspace/out_snapshot \
    -v `pwd`/dataset:/workspace/dataset \
    vietanhdev/openvi-image-classification:latest \
    python train.py --network resnet18 \
    --num_classes 2 \
    --input_size 224 \
    --epochs 20 \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --dataset dataset/mvtec \
    --path_pretrain pretrains/resnet18-5c106cde.pth \
    --job_name obj_classify
