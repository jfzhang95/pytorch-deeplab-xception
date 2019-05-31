python train.py --backbone mobilenet --lr 0.01 --workers 0 --epochs 40 --batch-size 3 --gpu-ids 0,1,2,3 --checkname deeplab-mobilenet-scarlet200 --eval-interval 1 --dataset scarlet200 $@

