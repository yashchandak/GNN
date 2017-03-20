python __main__.py  --path /home/priyesh/Desktop/Codes/Sample_Run/ \
                    --percent 25 \
                    --dataset cora \
                    --reduce 32 \
                    --hidden 16 \
                    --max_outer 100 \
                    --lu 0.5 \
                    --boot_epochs 4 \
                    --drop_in 0.5 \
                    --drop_out 0.75 \
                    --folder_suffix a1 | tee log25-a1.txt
#python __main__.py --path /home/priyesh/Desktop/Codes/Sample_Run/ --reduce 64 --percent 4 --folder_suffix 64 > log4-64.txt &