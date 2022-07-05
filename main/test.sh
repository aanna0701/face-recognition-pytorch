for w in /workspace/cosmos-eskim-facerecognition-shlee/save/train/6-27_6h14m-29s_AlterNet50_PartialFC_AdamW_lr_0.0005_0022/100_epoch_encoder.pth /workspace/cosmos-eskim-facerecognition-shlee/save/train/6-27_23h33m-15s_AlterNet50_PartialFC_AdamW_lr_0.0005_0022/100_epoch_encoder.pth /workspace/cosmos-eskim-facerecognition-shlee/save/train/6-28_1h18m-9s_AlterNet50_PartialFC_AdamW_lr_0.0005_0022/100_epoch_encoder.pth
do
python -u main.py --mode test --network AlterNet50 --test_type pair --ckpt_path ${w}
done

for w in /workspace/cosmos-eskim-facerecognition-shlee/save/train/6-27_3h9m-11s_ResNet50_PartialFC_AdamW_lr_0.0005/100_epoch_encoder.pth /workspace/cosmos-eskim-facerecognition-shlee/save/train/6-28_0h27m-0s_ResNet50_PartialFC_AdamW_lr_0.0005/100_epoch_encoder.pth /workspace/cosmos-eskim-facerecognition-shlee/save/train/6-28_2h11m-59s_ResNet50_PartialFC_AdamW_lr_0.0005/100_epoch_encoder.pth
do
python -u main.py --mode test --network ResNet50 --test_type pair --ckpt_path ${w}
done