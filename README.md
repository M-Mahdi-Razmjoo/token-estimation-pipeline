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
You can also run individual steps. For example, to only run the MLP model for the `r50k` tokenizer on `all` data scopes:
```bash
./scripts/mlp_models/r50k_all.bash
```

### 6. Outputs
Features are saved in the `features/` directory.

Model reports are saved in the `results/` directory.


