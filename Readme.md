<!-- ## Exp 1: Cross Entropy Loss

```bash
for dataset in milan cairo kyoto7
do
    for Bi in False True
    do
        for seed in 10 30 50
        do
            python ContrastLearning.py --dataset $dataset --BiLSTM $Bi --seed $seed --mode cross --weight 1
        done
    done
done
``` -->

<!-- ## Exp 2: Temporal Image /Spatial Image/ Concat Image

需要把第21行的“files=f'concat\\{dataset}\\{i}'“中的concat，分别修改为”loc_img“ "image_data"，然后每个数据集跑3个种子
```bash
for dataset in milan cairo kyoto7
do
    for seed in 10 30 50
    do
        python CNN_image.py --dataset $dataset --seed $seed
    done
done
``` -->

## Exp 3: Ablation on Sensor Color
需要把ContrastLearning.py第85行的“root_path=f'concat\\{dataset}\\{i}'“中的concat修改为”black“ 然后每个数据集跑3个种子
```bash
for dataset in milan cairo kyoto7
do
    for seed in 10 30 50
    do
        python ContrastLearning.py --dataset $dataset --BiLSTM False --seed $seed --mode cross
    done
done
``` 

## Exp 4: Ablation on Removal
需要把ContrastLearning.py
第85行的“root_path=f'concat\\{dataset}\\{i}'“中的concat修改为”wo_concat“ 
第79行的“root_path=f'seq\\{dataset}\\{i}'“中的seq修改为”wo_seq“ 
然后每个数据集跑3个种子
```bash
for dataset in milan cairo kyoto7
do
    for seed in 10 30 50
    do
        python ContrastLearning.py --dataset $dataset --BiLSTM False --seed $seed --mode cross
    done
done
``` 

## Exp 5: Ablation on frequency
需要把ContrastLearning.py
第85行的“root_path=f'concat\\{dataset}\\{i}'“中的concat修改为”fre005_concat“ 
第79行的“root_path=f'seq\\{dataset}\\{i}'“中的seq修改为”fre005_seq“ 
然后每个数据集跑3个种子
```bash
for dataset in milan kyoto7
do
    for seed in 20 40 80
    do,
        python ContrastLearning.py --dataset $dataset --BiLSTM False --seed $seed --mode cross
    done
done
``` 

然后把ContrastLearning.py
第85行的“root_path=f'concat\\{dataset}\\{i}'“中的concat修改为”fre01_concat“ 
第79行的“root_path=f'seq\\{dataset}\\{i}'“中的seq修改为”fre01_seq“ 
然后每个数据集跑3个种子
```bash
for dataset in kyoto7
do
    for seed in 20 40 80
    do
        python ContrastLearning.py --dataset $dataset --BiLSTM False --seed $seed --mode cross
    done
done
``` 

## Exp 6: Ablation on padding length
需要把ContrastLearning.py
第85行的“root_path=f'concat\\{dataset}\\{i}'“中的concat修改为”1000_concat“ 
第79行的“root_path=f'seq\\{dataset}\\{i}'“中的seq修改为”1000_seq“ 
然后每个数据集跑3个种子
```bash
for dataset in milan cairo kyoto7
do
    for seed in 10 30 50
    do
        python ContrastLearning.py --dataset $dataset --BiLSTM False --seed $seed --mode cross
    done
done
``` 

然后把ContrastLearning.py
第85行的“root_path=f'concat\\{dataset}\\{i}'“中的concat修改为”2000_concat“ 
第79行的“root_path=f'seq\\{dataset}\\{i}'“中的seq修改为”2000_seq“ 
然后每个数据集跑3个种子
```bash
for dataset in milan cairo kyoto7
do
    for seed in 10 30 50
    do
        for imgseq in 0.3 0.5 1.5 2.0
        do
            python ContrastLearning.py --dataset $dataset --seed $seed --imgseq $imgseq --BiLSTM True
        done
    done
done
``` 

<!-- ## Exp 7: Extra LSTM
```bash
for dataset in milan cairo kyoto7
do
    for seed in 10 30 50
    do
        python LSTM.py --dataset $dataset --BiLSTM False --seed $seed
    done
done
```  -->


```bash
for dataset in milan cairo kyoto7
do
    for seed in 10 30 50
    do
        for BiLSTM in True False
        do
            python ContrastLearning.py --dataset $dataset --BiLSTM $BiLSTM --seed $seed --mode single
        done
    done
done
``` 