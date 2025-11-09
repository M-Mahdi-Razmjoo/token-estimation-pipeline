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

This reads raw data from `data/`, adds new tokenizer columns (`deepseek_r1`, `qwen_qwq`), and saves the result in `data_enriched/`.
```bash
./scripts/enrich_data.bash
```

Next, build the features:

This uses the cleaned data to create feature files in the `features/` directory.
```bash
./scripts/build_features.bash
```

Finally, you can run the specific model training script:
```bash
./scripts/rf_models/qwen_qwq_all.bash
```


### 6. Outputs
`data_enriched/`: Contains the original data enriched with new tokenizer columns. This is the new source for all subsequent steps.

`features/`: Contains data enriched with engineered features, ready for model training.

`results/`: Contains the final model performance reports and training logs.


