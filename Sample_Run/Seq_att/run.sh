python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 1 \
                    --dataset facebook \
                    --reduce 0 \
                    --hidden 16 \
                    --wce False \
                    --concat False \
                    --max_outer 15 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.25 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceF | tee log-fb1-wceF.txt

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
