MODEL_NAME='f30k_i2t'
GPU_DEVICES='1'
DATASET_NAME='f30k'
DATA_PATH='./data_glove/'${DATASET_NAME}
VOCAB_PATH='./data_glove/vocab_GloVe'

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
 --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
 --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES}\
 --num_epochs=30 --lr_update=20 --learning_rate=.0005 --workers 10 \
 --log_step 100 --embed_size 1024 --vse_mean_warmup_epochs 3 --trans_step 1 \
 --attn_type i2t --t2i_smooth 10.0 --i2t_smooth 9.0 --batch_size 120
