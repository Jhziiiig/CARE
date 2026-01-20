for dataset in kyoto7
do
    for Bi in 1
    do
        for seed in 10 30 50
        do
            for filter_t in sens
            do
                for filter_threshold in 0.1
                do
                    for weight in 0.1 0.3  0.5
                    do
                    if [ "$filter_threshold" == "0.00" ] && [ "$filter" == "room" ]; then
                        continue
                    fi
                    if [ -f model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.}/Test.json ]; then
                        echo "model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.} exists"
                    else
                        echo "model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.} does not exist"
                        CUDA_VISIBLE_DEVICES=2 python ContrastLearning.py --dataset $dataset --BiLSTM $Bi --seed $seed --filter $filter_t --filter_threshold $filter_threshold --weight $weight --full-training
                    fi
                    done
                done
            done
        done
    done
done

# for dataset in milan cairo
# do
#     for Bi in 1
#     do
#         for seed in 10 30 50
#         do
#             for filter_t in sens
#             do
#                     for filter_threshold in 0.01
#                     do
#                     if [ "$filter_threshold" == "0.00" ] && [ "$filter" == "room" ]; then
#                         continue
#                     fi
#                     if [ -f model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.}/Test.json ]; then
#                         echo "model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.} exists"
#                     else
#                         echo "model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.} does not exist"
#                         CUDA_VISIBLE_DEVICES=4 python ContrastLearning.py --dataset $dataset --BiLSTM $Bi --seed $seed --filter $filter_t --filter_threshold $filter_threshold --weight=0.7
#                     fi
#                     done
#             done
#         done
#     done
# done


# for dataset in kyoto7
# do
#     for Bi in 1
#     do
#         for seed in 10 30 50
#         do
#             for filter_t in sens
#             do
#                     for filter_threshold in 0.1
#                     do
#                     if [ "$filter_threshold" == "0.00" ] && [ "$filter" == "room" ]; then
#                         continue
#                     fi
#                     if [ -f model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.}/Test.json ]; then
#                         echo "model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.} exists"
#                     else
#                         echo "model/$dataset-$Bi-$seed-cross-scl_filter_${filter_t}_t${filter_threshold#0.} does not exist"
#                         CUDA_VISIBLE_DEVICES=4 python ContrastLearning.py --dataset $dataset --BiLSTM $Bi --seed $seed --filter $filter_t --filter_threshold $filter_threshold --weight=0
#                     fi
#                     done
#             done
#         done
#     done
# done
