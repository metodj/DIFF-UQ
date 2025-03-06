path="imagenet256_uvit_huge.py"
steps="50"
DIS="time_uniform"
mc_size="5"
sample_batch_size="128"
total_n_sample="12032"
train_la_data_size="50"  # the size of dataset used for Laplace approximation: size_of_total_dataset//train_la_data_size
fixed_class="10000"  # all classes are generated if fixed_class="10000" else range from 0 to 999
encoder_path="..."
uvit_path="..."
DEVICES=0
seed=1234
exp_path="./images"

echo "Running for seed $seed"
CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
--config $path --timesteps=$steps --eta 0 --skip_type=$DIS --train_la_batch_size 32 \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --encoder_path=$encoder_path --uvit_path=$uvit_path --seed=$seed --exp_path=$exp_path
