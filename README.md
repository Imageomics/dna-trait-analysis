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
    - Copy that file and create a new one. Then update ```DEFAULT_YAML_CONFIG_PATH``` in ```src/gtp/configs/loaders.py```

# Run Tests
To run tests, run the follwing command:
```
pytest tests
```

NOTE: these tests depend on the current configs and may need to be adjusted for your development environment.

# Preprocess Data
To preprocess the Data, you can either:
- Run the notebook in ```notebooks/preprocess_pipeline.ipynb```.
- Or, run the commands:
```
python -m gtp.pipelines.preprocess --method phenotype
python -m gtp.pipelines.preprocess --method genotype
```

NOTE: preprocessing genotype data takes a long time. You can add the ```num-processes``` argument to the command to help speed this up. 4 by default. Do not be fooled by the initial processing speed as smaller files are processed first.

# Create Data Splits
To keep data splits consistent across runs, precompute the splits with the following command:
```
python -m gtp.pipelines.create_training_splits
```

# Training the Model
Run the following command:
```
python -m gtp.train_whole_genome --chromosome 18 --species erato --color color_3 --wing forewings --epochs 100 --batch_size 32 --exp_name test
```