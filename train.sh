
# Todo dataset_mode Jittor hist_loss
#input_path="/home/gavin/Documents/Dataset/Jittor2023/train_resized"
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--name histlosstest --dataset_mode Jittor \
#--label_dir $input_path/labels \
#--image_dir $input_path/imgs \
#--use_vae --use_hist \
#--label_nc 29 --no_instance  --batchSize 10 \
#--continue_train --which_epoch 190

# Todo baseline resume
#input_path="/home/gavin/Documents/Dataset/Jittor2023/train_resized"
#CUDA_VISIBLE_DEVICES=1 python train.py \
#      --name basic \
#      --dataset_mode custom \
#      --label_dir $input_path/labels \
#      --image_dir $input_path/imgs \
#      --label_nc 29  --no_instance  --batchSize 12 \
#      --lr 0.0001 \
#      --continue_train --which_epoch 190

# Todo Gaugan2
input_path="/home/gavin/Documents/Dataset/Jittor2023/train_resized"
CUDA_VISIBLE_DEVICES=1 python train.py \
      --name poe_gan \
      --model PoEGAN \
      --dataset_mode Jittor \
      --label_dir $input_path/labels \
      --image_dir $input_path/imgs \
      --label_nc 29  --no_instance  --batchSize 12