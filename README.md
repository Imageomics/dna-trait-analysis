# dna-trait-analysis
Goal: to find associations between dna data and visual traits.

# Installation
```
conda create -n gtp python=3.10
conda activate gtp
pip install -e .
```

# Set Options
To personalize options can either:
    - Change the configurations in `src/gtp/configs/default_configs.yaml`
    - Copy that file and create a new one. Then update 

# Preprocess Data
Run the notebook in ```notebooks/preprocess_pipeline.ipynb```. You may have to change the input and output directory constants and even the ```DNA_SCOPE``` depending on your task.

# Training the Model
Run the following command:
```
python -m gtp.train_whole_genome --chromosome 18 --species erato --color color_3 --wing forewings --epochs 100 --batch_size 32 --exp_name test
```