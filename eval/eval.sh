EXP="/nvmestore/mjazbec/diffusion/bayes_diff/exp_repo_clean/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234"
H=128
# H=256

m=0
unc_name="entropy_clip"
reverse="false"

echo "Computing FID, precision, recall for Random baseline"
python fid.py calc --images="${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num 10000 --fid_features="${EXP}/${m}/fid_features.pt"
python precision_recall_torch.py --ref ./precision-recall-refs/image_net_val_${H}_fid_features.pt --eval "${EXP}/${m}/fid_features.pt"

echo "Computing FID, precision, recall for filtered samples based on generative uncertainty"
python idx_sort.py --path ${EXP} --name ${unc_name} --N 10000 --reverse ${reverse}
python fid.py calc --images="${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --fid_features="${EXP}/${m}/fid_features_filtered_${unc_name}.pt" --idx_path="${EXP}/idx_sorted_10000_${unc_name}.npy"
python precision_recall_torch.py --ref ./precision-recall-refs/image_net_val_${H}_fid_features.pt --eval "${EXP}/${m}/fid_features_filtered_${unc_name}.pt"


