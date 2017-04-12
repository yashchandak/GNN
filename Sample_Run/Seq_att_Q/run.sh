python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 25 \
                    --dataset cora \
                    --reduce 32 \
                    --hidden 16 \
                    --concat False \
                    --max_outer 100 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTMgated \
                    --attention 0 \
                    --folder_suffix gated | tee log-cora25-gated.txt


#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
