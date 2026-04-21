EXPID=$(date +"%Y%m%d_%H%M%S")

NUM_GPU=1
python train.py \
--config 'configs/train.yaml' \
--output_dir './results' \
--rank 0 \
--log_num ${EXPID} \
--world_size $NUM_GPU \
