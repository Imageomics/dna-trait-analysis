# dna-trait-analysis
Goal: to find associations between dna data and visual traits.

# Installation
```
uv sync
```

# Set Options
To personalize options can either:
    - Change the configurations in `src/gtp/configs/default.yaml`
    - Copy that file and create a new one. Then update ```DEFAULT_YAML_CONFIG_PATH``` in ```src/gtp/configs/loaders.py```

# Run Tests
To run tests, run the follwing command:
```
uv run pytest tests
```

NOTE: these tests depend on the current configs and may need to be adjusted for your development environment.

# Preprocess Data
To preprocess the Data, you can either:
- Run the notebook in ```notebooks/preprocess_pipeline.ipynb```.
- Or, run the commands:
```
uv run python -m gtp.pipelines.preprocess --method phenotype
uv run python -m gtp.pipelines.preprocess --method genotype
```

NOTE: preprocessing genotype data takes a long time. You can add the ```num-processes``` argument to the command to help speed this up. 4 by default. Do not be fooled by the initial processing speed as smaller files are processed first.

# Create Data Splits
To keep data splits consistent across runs, precompute the splits with the following command:
```
uv run python -m gtp.pipelines.create_training_splits
```

# Training the Model
Run the following command:
```
uv run python -m gtp.pipelines.train --chromosome 18 --species erato --color color_3 --wing forewings --epochs 100 --batch_size 32 --exp_name debug
```

# Calculate Attributions
To calculate the attributions of a single chromosome or entire genome (depending on configs: see `experiment.genotype_scope`), run the following command:
```
uv run python -m gtp.pipelines.process_attributions --species erato --color color_3 --wing forewings --exp_name debug
```

NOTE: you have to have trained a model for every chromosome for a species in order for the above command to work on the entire genome. Otherwise, it'll only plot / calculate data for chromosomes with a trained model.

# Plot Attributions
To plot the attributions of a single chromosome or entire genome (depending on configs: see `experiment.genotype_scope`), run the following command:
```
uv run python -m gtp.pipelines.plot_attributions --species erato --color color_3 --wing forewings --exp_name debug
```

NOTE: you have to have trained a model for every chromosome for a species in order for the above command to work on the entire genome. Otherwise, it'll only plot / calculate data for chromosomes with a trained model.

# Calculate Pair-wise Epistasis Interactions
Run the following command to get potential epistasis interactions:
```
TODO
```

# Extracting Projection Matrix from .rds data file
To extract the PCA projection matrix from ```.rds``` files as a result of the ```genotype-phenotype``` pipeline, use the ```scripts/r/extract_projection_matrix.R``` R script. You will have to install R on your machine. With conda it can be done by running: ```conda install conda-forge::r-base```. Run the script as follows:
```
Rscript scripts/r/extract_projection_matrix.R [input_file].rds [output_file].csv
```