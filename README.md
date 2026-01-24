# CARE
This is the offical repo of CARE: Contrastive Alignment for ADL Recognition from Event-Triggered Sensor Streams, an end-to-end framework that jointly optimizes representation learning via Sequence-Image Contrastive Alignment (SICA) and classification via cross-entropy, ensuring both cross-representation alignment and task-specific discriminability.
## Setup
Our code is working on Python 3.9. You can run the following code to setup.
```bash
conda env create -f environment.yaml
conda activate HAR
```
## Data Download
Data can be either downloaded from CASAS(https://casas.wsu.edu/) or our google drive(https://drive.google.com/drive/folders/1eWQihhGHWFVzopOJw_xP7Yxl8YURE6l1?usp=sharing). We trained and tested on Milan, Cairo and Kyoto7.

## Data Preprocessing
Before the model training, please run the following code to process your raw data.

If you want to use our default setting, you can directly run:
```bash
bash dataset_gen.sh
```
If you want to customize the padding length, dataset, frequency filtering threshold or temporal binning, you can run:
```python
python DataGeneration.py --datasets milan  --mode sensor --maxlen 100 --threshold 0.01 --off
python DataGeneration.py --datasets cairo  --mode sensor --threshold 0.01 --maxlen 100 --off
python DataGeneration.py --datasets kyoto7  --mode sensor --threshold 0.1 --maxlen 100 --off
```

## Model Training and Testing
Run the following code can quickly get the results:
If you use our default setting to process the data, then you can just run:
```bash
bash run.sh
```
If you customize your setting, please make sure the following setting are consistent:
```bash
for dataset in kyoto7
do
    for Bi in 1
    do
        for seed in 10 
        do
            for filter_threshold in 0.1
            do
                for weight in 0.5
                do
                echo "model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.} does not exist"
                CUDA_VISIBLE_DEVICES=4 python ContrastLearning.py --dataset $dataset --BiLSTM $Bi --seed $seed --filter $filter_t --filter_threshold $filter_threshold --weight $weight --full-training --maxlen 1500
                fi
                done
            done
        done
    done
done
```

All the results will be recorded in the local logs.
## Model Tuning
If you want to determine some hyperparameters by K-fold cross validation, please remove the `--full-training` option in the training command. The default K is 5 and it can be set by `--K`.
