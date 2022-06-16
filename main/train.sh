for m in AlterNet50
do
for r in 0.3
do

# python train.py --sample_rate ${r} --optimizer SGD --network ${m} --lr 0.1

for l in 5e-4
do
CUDA_VISIBLE_DEVICES='0' python -u main.py --mode train --sample_rate ${r} --optimizer AdamW --network ${m} --lr ${l}
done

done
done
