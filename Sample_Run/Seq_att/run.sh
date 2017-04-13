python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 6 \
                    --folds 1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --reduce 0 \
                    --hidden 10 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 10 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.25 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-fb6-wceT.txt &


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 7 \
                    --folds 1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --reduce 0 \
                    --hidden 10 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 10 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.25 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-fb7-wceT.txt


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 8 \
                    --folds 1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --reduce 0 \
                    --hidden 10 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 10 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.25 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-fb8-wceT.txt &


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 9 \
                    --folds 1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --reduce 0 \
                    --hidden 10 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 10 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.25 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-fb9-wceT.txt


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 10 \
                    --folds 1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --reduce 0 \
                    --hidden 10 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 10 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.25 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-fb10-wceT.txt &


#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
