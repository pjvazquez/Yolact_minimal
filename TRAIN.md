# Training process in p3.2xlarge instance in aws

## 2021-05-14 init train

- need to install
pip install tensorboardX
pip install pycocotools
pip install terminaltables

### Train with resnet101 backbone on one GPU with a batch size of 8 (default).
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --train_bs=8

broke in 15000 iteration

nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --train_bs=8 --resume weights/latest_res101_coco_15000.pth