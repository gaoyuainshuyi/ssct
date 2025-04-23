MODEL_NAME='coco_i2t'
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data_glove/'${DATASET_NAME}
VOCAB_PATH='./data_glove/vocab_GloVe'

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
 --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
 --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES}\
 --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 10 \
 --log_step 100 --embed_size 1024 --vse_mean_warmup_epochs 5 --trans_step 1 \
 --attn_type i2t --t2i_smooth 10.0 --i2t_smooth 10.0 --batch_size 120 --resume runs/coco_i2t/checkpoint_1.pth 
