# Token Estimation Pipeline
A complete pipeline to train and evaluate models for token estimation.

**Important:** All commands must be run from the project's root directory.

### 1. Setup
```bash
git clone https://github.com/M-Mahdi-Razmjoo/token-estimation-pipeline.git
cd token-estimation-pipeline
```

### 2. Add Data
Place your `merged_*.parquet` files into the `data/` directory. This folder is already created.


### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Grant Permissions
Make the run scripts executable
```bash
chmod +x scripts/**/*.bash scripts/*.bash
```

### 5. Run Pipeline
#### Run the entire pipeline:
This command executes all steps sequentially.
```bash
./scripts/run_all.bash
```

#### Run a specific step:
First, run the data preprocessing step (if not already done):

This creates the cleaned data in the `data_filtered/` directory.
```bash
./scripts/preprocess_data.bash
```

Next, build the features:

This uses the cleaned data to create feature files in the `features/` directory.
```bash
./scripts/build_features.bash
```

Finally, you can run the specific model training script:
```bash
./scripts/mlp_models/r50k_all.bash
```


### 6. Outputs
`data_filtered/`: Contains the cleaned data after the preprocessing step.

`features/`: Contains data enriched with engineered features, ready for model training.

`results/`: Contains the final model performance reports and training logs.


