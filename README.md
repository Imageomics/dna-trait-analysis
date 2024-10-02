# dna-trait-analysis
Goal: to find associations between dna data and visual traits.

# Installation
```
pip install -r requirements.txt
pip install -e .
```

# Preprocess Data
Run the notebook in ```notebooks/run_pipeline.ipynb```. You may have to change the input and output directory constants and even the ```DNA_SCOPE``` depending on your task.

# Training the Model
Run the following command:
```
python -m gtp.train_whole_genome --chromosome 18 --species erato --color color_3 --wing forewings --epochs 100 --batch_size 32 --exp_name test
```