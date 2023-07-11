1. Pretrain the model using the source dataset:   
  python ./pretrain_new.py --ls --source_dataset miniImageNet  --backbone resnet10  --model base --tag default

2. Re-train the model using the unlabeled target dataset: \
  python ./pretrain_new.py --pls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10   --fre 50 --alpha 1e-6 --model byol  --pr 0.3

3. Fine-tune the trained model and evaluate the final model:   
  python ./finetune_new.py --pls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts head --split_seed 1 --pr 0.3 --fre 50



# Soft Weight Pruning for Cross-Domain Few-Shot Learning with Unlabeled Target Data
The checkpoints of final models can be downloaded via the following link:
https://drive.google.com/file/d/1rFpCUqZoPDXiWy5itsBCo7TxA2WsSONr/view?usp=sharing
