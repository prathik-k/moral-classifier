# Steps for model training/evaluation:

1. Create a conda environment with the packages listed in environment.yml with this command: conda env create -f environment.yml
2. Ensure that there are external folders with these relative paths (with the allowed model names being BERT,ALBERT or ROBERTA): "../dataloaders/$MODEL_NAME$" and "../trained_models/$MODEL_NAME$"
3. To generate the required dataloaders, navigate within the directory of each model, and execute this statement: python preprocess_data.py
4. Again, navigate to the folder with each model, and execute this statement to train the models: python train_and_eval.py
5. Rerun the above statement in order to generate test metrics. All the results for the tests will be saved in the "../trained_models/$MODEL_NAME$/".

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


