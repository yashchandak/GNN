python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project path_attn \
                    --percent 6 \
                    --labels labels_dfs \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --gradients 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 20 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wce_batch | tee dump/log-cora1-wce_batch.txt

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
