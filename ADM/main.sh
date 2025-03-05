DEVICES="0"
data="imagenet128_guided"
steps="50"
mc_size="5"
sample_batch_size="256"
total_n_sample="12032"
train_la_data_size="100"
train_la_batch_size="32"
DIS="uniform"
fixed_class="10000"
seed="1234"
exp_path="/nvmestore/mjazbec/diffusion/bayes_diff/exp_repo_clean"

echo "Generating samples"
CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
--config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size $train_la_batch_size \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --fixed_class=$fixed_class --seed=$seed --exp_path=$exp_path
