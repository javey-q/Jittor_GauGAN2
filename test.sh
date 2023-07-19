input_path=../dataset/Jittor/val_A_labels_resized
img_path=../dataset/Jittor/train_resized/imgs
load_epoch=200
CUDA_VISIBLE_DEVICES="0" python test.py  \
--name basic --dataset_mode Jittor\
--label_dir $input_path \
--image_dir $img_path \
--which_epoch $load_epoch \
--label_nc 29 --no_instance --use_vae --no_pairing_check