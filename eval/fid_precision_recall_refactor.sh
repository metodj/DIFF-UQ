# ROOT="/nvmestore/mjazbec/diffusion/bayes_diff/exp_final/IMAGENET128"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_1234"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_1111"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_2222"
# EXP="ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_epi_unc_1111"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_epi_unc_2222"
# H=128

# # ROOT="/nvmestore/mjazbec/diffusion/bayes_diff/exp/IMAGENET128"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_ale_unc_debug"
# ROOT="/nvmestore/mjazbec/diffusion/bayes_diff/exp_final/IMAGENET128"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_ale_unc_1234"
# # EXP="ddim_fixed_class10000_train%100_step50_S5_ale_unc_1111"
# EXP="ddim_fixed_class10000_train%100_step50_S5_ale_unc_2222"
# H=128

ROOT="/nvmestore/mjazbec/diffusion/bayes_diff/exp_repo_clean/IMAGENET128"
EXP="ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234"
H=128

# ROOT="/nvmestore/mjazbec/diffusion/bayes_diff/exp/imagenet256"
# # EXP="dpmUQ_fixed_class10000_train%50_step50_S10_1233"
# # EXP="dpmUQ_fixed_class10000_train%50_step50_S10_epi_unc_1233"
# # EXP="dpmUQ_fixed_class10000_train%50_step50_S5_epi_unc_1111"
# EXP="dpmUQ_fixed_class10000_train%50_step50_S5_epi_unc_2222"
# # EXP="dpmUQ_fixed_class10000_train%50_step50_S5_1111"
# # EXP="dpmUQ_fixed_class10000_train%50_step50_S5_2222"
# H=256

N=12032

# m="samples"
# unc_name="all_unc"
# reverse="false"

# m=0
# # unc_name="cos_sim_clip"
# unc_name="cos_sim_clip_2"
# # unc_name="cos_sim_clip_6"
# reverse="true"

m=0
# m="best"
# unc_name="entropy_clip_2"
# unc_name="paide_clip_11"
# unc_name="entropy_clip_6"
unc_name="entropy_clip"
# unc_name="entropy_clip_6_debug"
# unc_name="paide_clip_6"
# unc_name="paide_clip_2"
# unc_name="all_std"
reverse="false"
# reverse="true"

python fid.py calc --images="${ROOT}/${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num 10000 --fid_features="${ROOT}/${EXP}/${m}/fid_features.pt"
# python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features.pt"
python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_${H}_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features.pt"

python idx_sort.py --path ${ROOT}/${EXP} --name ${unc_name} --N 10000 --reverse ${reverse}
python fid.py calc --images="${ROOT}/${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${ROOT}/${EXP}/${m}/fid_features_filtered_${unc_name}.pt" --idx_path="${ROOT}/${EXP}/idx_sorted_10000_${unc_name}.npy"
# python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features_filtered_${unc_name}.pt"
python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_${H}_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features_filtered_${unc_name}.pt"


### filtering out 2 samples per class (class_distribution.py)
# python fid.py calc --images="${ROOT}/${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${ROOT}/${EXP}/${m}/fid_features_filtered_${unc_name}_manual.pt" --idx_path="${ROOT}/${EXP}/selected_idx.npy"
# python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features_filtered_${unc_name}_manual.pt"

# ### CLIP baseline
# python fid.py calc --images="${ROOT}/${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${ROOT}/${EXP}/${m}/fid_features_filtered_clip_baseline.pt" --idx_path="${ROOT}/${EXP}/idx_sorted_10000_CLIP_baseline.npy"
# python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features_filtered_clip_baseline.pt"


##### combined
# python fid.py calc --images="${ROOT}/${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${ROOT}/${EXP}/${m}/fid_features_filtered_realism_ours.pt" --idx_path="${ROOT}/${EXP}/idx_sorted_10000_realism_ours.npy"
# python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_${H}_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features_filtered_realism_ours.pt"
# python fid.py calc --images="${ROOT}/${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${ROOT}/${EXP}/${m}/fid_features_filtered_realism_rarity.pt" --idx_path="${ROOT}/${EXP}/idx_sorted_10000_realism_rarity.npy"
# python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_${H}_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features_filtered_realism_rarity.pt"
# python fid.py calc --images="${ROOT}/${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${ROOT}/${EXP}/${m}/fid_features_filtered_rarity_ours.pt" --idx_path="${ROOT}/${EXP}/idx_sorted_10000_rarity_ours.npy"
# python precision_recall_torch.py --ref /ivi/xfs/mjazbec/edm/datasets/image_net_val_${H}_fid_features.pt --eval "${ROOT}/${EXP}/${m}/fid_features_filtered_rarity_ours.pt"

