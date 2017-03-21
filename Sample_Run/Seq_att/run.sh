python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 20 \
                    --dataset MLGene \
                    --reduce 0 \
                    --hidden 16 \
                    --max_outer 10 \
                    --lu 0.5 \
                    --boot_epochs 4 \
                    --drop_in 0 \
                    --drop_out 0.2 \
                    --cell RNN \
                    --attention 2 \
                    --folder_suffix a2 | tee log-PPI20-a2.txt

#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &

: <<'Batch_Comment'
comment section
Batch_Comment
