python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project path_attn \
                    --percent 25 \
                    --labels labels_random \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --batch_size 256 \
                    --gradients 1 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 50 \
                    --max_depth 5 \
                    --max_inner 15 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --gradients 1 \
                    --folder_suffix wce_batch | tee dump/log-cora25-wce_batch.txt

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
