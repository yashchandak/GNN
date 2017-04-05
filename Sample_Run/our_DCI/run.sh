python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 25 \
                    --dataset facebook \
                    --reduce 0 \
                    --hidden 16 \
                    --concat False \
                    --max_outer 1000 \
                    --lu 1.0 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.00001 \
                    --boot_epochs 4 \
                    --drop_in 0.2 \
                    --drop_out 0.2 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix DCI2 | tee log-cora25-DCI2.txt

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
