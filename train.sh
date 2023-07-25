#input_path=../dataset/Jittor/train_resized
#CUDA_VISIBLE_DEVICES="0" python train.py \
#--name basic --dataset_mode Jittor \
#--label_dir $input_path/labels \
#--image_dir $input_path/imgs \
#--label_nc 29 --no_instance --use_vae --batchSize 4
input_path="/home/gavin/Documents/Dataset/Jittor2023/train_resized"
CUDA_VISIBLE_DEVICES=1 python train.py \
      --name basic \
      --dataset_mode custom \
      --label_dir $input_path/labels \
      --image_dir $input_path/imgs \
      --label_nc 29  --no_instance  --batchSize 12 \
      --lr 0.0001 \
      --continue_train --which_epoch 190

input_path="/home/gavin/Documents/Dataset/Jittor2023/train_resized"
CUDA_VISIBLE_DEVICES=1 python train.py \
      --name basic \
      --netG PoE \
      --netD PoE \
      --dataset_mode custom \
      --label_dir $input_path/labels \
      --image_dir $input_path/imgs \
      --label_nc 29  --no_instance  --batchSize 12 \
      --lr 0.0001 \
      --continue_train --which_epoch 190