Training example：
python code/Thc-net.py --data_dir "data/6_cell_input_updated/6_cell_input_updated_100kb/" --task "cla" --model "transformer" -Ts --epoch 10 --resolution "100kb" --cross_validation True --add_mean_evec True --num_fold 5 --special_tag "test" --cell "GM12878" --learning_rate 0.01 --hidden 128 --layer 3 --n_heads 4 --test

