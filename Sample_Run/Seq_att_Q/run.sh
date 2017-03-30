python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 25 \
                    --dataset cora \
                    --reduce 0 \
                    --hidden 16 \
                    --concat False \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.00001 \
                    --boot_epochs 4 \
                    --drop_in 0.25 \
                    --drop_out 0.6 \
                    --cell RNN \
                    --attention 2 \
                    --folder_suffix la0r0lu2rmsbtest | tee log-cora25-la0r0lu2rmsbtest.txt

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
