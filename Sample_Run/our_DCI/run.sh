python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 1 \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --wce 1 \
                    --add_degree 1 \
                    --reduce 0 \
                    --hidden 16 \
                    --concat 0 \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --drop_in .5 \
                    --drop_out .75 \
                    --cell LSTM \
                    --folder_suffix wce | tee dump/log-cora-wce1.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 2 \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --wce 1 \
                    --add_degree 1 \
                    --reduce 0 \
                    --hidden 16 \
                    --concat 0 \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --drop_in .5 \
                    --drop_out .75 \
                    --cell LSTM \
                    --folder_suffix wce | tee dump/log-cora-wce2.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 3 \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --wce 1 \
                    --add_degree 1 \
                    --reduce 0 \
                    --hidden 16 \
                    --concat 0 \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --drop_in .5 \
                    --drop_out .75 \
                    --cell LSTM \
                    --folder_suffix wce | tee dump/log-cora-wce3.txt &


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 4 \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --wce 1 \
                    --add_degree 1 \
                    --reduce 0 \
                    --hidden 16 \
                    --concat 0 \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --drop_in .5 \
                    --drop_out .75 \
                    --cell LSTM \
                    --folder_suffix wce | tee dump/log-cora-wce4.txt &


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 5 \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --wce 1 \
                    --add_degree 1 \
                    --reduce 0 \
                    --hidden 16 \
                    --concat 0 \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --drop_in .5 \
                    --drop_out .75 \
                    --cell LSTM \
                    --folder_suffix wce | tee dump/log-cora-wce5.txt &


python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --project our_DCI \
                    --percent 6 \
                    --folds 1_2_3_4_5 \
                    --dataset cora \
                    --wce 1 \
                    --add_degree 1 \
                    --reduce 0 \
                    --hidden 16 \
                    --concat 0 \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.01 \
                    --opt adam \
                    --l2 0.00001 \
                    --drop_in .5 \
                    --drop_out .75 \
                    --cell LSTM \
                    --folder_suffix wce | tee dump/log-cora-wce6.txt &





#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
