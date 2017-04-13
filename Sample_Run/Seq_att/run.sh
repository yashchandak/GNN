python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 1 \
                    --labels labels_dfs \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-cora1-wceT.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 2 \
                    --labels labels_dfs \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-cora2-wceT.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 3 \
                    --labels labels_dfs \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-cora3-wceT.txt

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 4 \
                    --labels labels_dfs \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-cora4-wceT.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 5 \
                    --labels labels_dfs \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-cora5-wceT.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 6 \
                    --labels labels_dfs \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-cora6-wceT.txt



#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
