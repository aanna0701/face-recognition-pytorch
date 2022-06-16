for m in AlterNet50 ResNet50
do
for r in 1
do

# python train.py --sample_rate ${r} --optimizer SGD --network ${m} --lr 0.1

for l in 5e-4
do
python train.py --sample_rate ${r} --optimizer AdamW --network ${m} --lr ${l}
done

done
done
