python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 202 \
                    --labels labels_mix \
                    --dataset citeseer \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 1 \
                    --concat 0 \
                    --max_outer 10 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceT | tee dump/log-citeseer202-wceT.txt &

python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 202 \
                    --labels labels_mix \
                    --dataset citeseer \
                    --reduce 0 \
                    --hidden 16 \
                    --wce 0 \
                    --concat 0 \
                    --max_outer 10 \
                    --lu 0.2 \
                    --lr 0.001 \
                    --opt adam \
                    --l2 0.001 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --cell LSTM \
                    --attention 0 \
                    --folder_suffix wceF | tee dump/log-citeseer202-wceF.txt &

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
