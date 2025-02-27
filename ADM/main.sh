# ddimUQ.py's IMAGENET128
DEVICES="0"
data="imagenet128_guided"
steps="50"
mc_size="5"
sample_batch_size="256"
total_n_sample="12032"
train_la_data_size="100"
DIS="uniform"
fixed_class="10000"
# uncertainty_type="bayes_diff"
uncertainty_type="epistemic"
# uncertainty_type="aleatoric"

# for seed in 1234 1111 2222; do
# for seed in 1111 2222; do
for seed in 1234; do
    echo "Running for seed $seed"
    CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
    --config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size 32 \
    --mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
    --total_n_sample=$total_n_sample --fixed_class=$fixed_class --seed=$seed --uncertainty_type=$uncertainty_type
done