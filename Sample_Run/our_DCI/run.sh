python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 9 \
                    --folds 0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --wce True \
                    --add_degree True \
                    --reduce 0 \
                    --hidden 10 \
                    --concat False \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --boot_epochs 4 \
                    --drop_in 0.2 \
                    --drop_out 0.2 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix DCI2deg | tee log-fb-DCIdeg9.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 6 \
                    --folds 0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --wce True \
                    --add_degree True \
                    --reduce 0 \
                    --hidden 10 \
                    --concat False \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --boot_epochs 4 \
                    --drop_in 0.2 \
                    --drop_out 0.2 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix DCI2deg | tee log-fb-DCIdeg6.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 7 \
                    --folds 0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --wce True \
                    --add_degree True \
                    --reduce 0 \
                    --hidden 10 \
                    --concat False \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --boot_epochs 4 \
                    --drop_in 0.2 \
                    --drop_out 0.2 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix DCI2deg | tee log-fb-DCIdeg7.txt &


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 8 \
                    --folds 0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --wce True \
                    --add_degree True \
                    --reduce 0 \
                    --hidden 10 \
                    --concat False \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --boot_epochs 4 \
                    --drop_in 0.2 \
                    --drop_out 0.2 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix DCI2deg | tee log-fb-DCIdeg8.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 5 \
                    --folds 0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16 \
                    --dataset facebook \
                    --wce True \
                    --add_degree True \
                    --reduce 0 \
                    --hidden 10 \
                    --concat False \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --boot_epochs 4 \
                    --drop_in 0.2 \
                    --drop_out 0.2 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix DCI2deg | tee log-fb-DCIdeg5.txt &

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
