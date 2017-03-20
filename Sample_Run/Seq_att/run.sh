python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 25 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --max_outer 10 \
                    --lu 0.75 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 1 \
                    --folder_suffix aD1 | tee log-cora4-aD1.txt

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment