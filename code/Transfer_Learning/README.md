# Using ALBERT for Transfer Learning

## Dataset preparation
Use
```
python download_glue_data.py --data_dir glue_data --tasks SST
```

## Evaluation 
```
python run_glue.py   --model_name_or_path albert-base-v2   --task_name SST-2   --data_dir glue_data/SST-2   --max_seq_length 128   --per_gpu_train_batch_size 64   --learning_rate 2e-5   --num_train_epochs 3.0 --output_dir output/ --model_type albert --config_name albert-base-v2 --tokenizer_name albert-base-v2 --do_eval 
```

## Fine-Tuning
```
python run_glue.py   --model_name_or_path albert-base-v2   --task_name SST-2    --data_dir glue_data/SST-2   --max_seq_length 128   --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0 --output_dir output/ --model_type albert --config_name albert-base-v2 --tokenizer_name albert-base-v2 --do_train --do_eval
```



