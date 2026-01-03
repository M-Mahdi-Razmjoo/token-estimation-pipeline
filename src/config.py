import os

BASE_PATH = "data"    #specify the base path where the parquet files are located
ENRICHED_PATH = "data_enriched"
FEATURES_PATH = "features"
RESULTS_PATH = "results"

FILE_NAMES = [f"merged_{i}.parquet" for i in range(1, 11)]
INPUT_FILES = [os.path.join(BASE_PATH, fname) for fname in FILE_NAMES]
ENRICHED_INPUT_FILES = [os.path.join(ENRICHED_PATH, f"enriched_{fname}") for fname in FILE_NAMES]

CONTENT_COLUMN = 'content'
LANGUAGE_COLUMN = 'language'
TOKENIZER_COLUMNS = {
    'r50k': 'tiktoken_r50k_base_len',
    'cl100k': 'tiktoken_cl100k_base_len',
    'o200k': 'tiktoken_o200k_base_len',
    'deepseek_r1': 'deepseek_r1',
    'qwen_qwq': 'qwen_qwq' ,
    'llama3_1_8b': 'llama3_1_8b'
}

LANGUAGE_SCOPES = ['all', 'english']

CV_FOLDS = 10

ALL_POTENTIAL_FEATURES = [
    'char_count', 'word_count', 'whitespace_count', 'has_non_ascii', 
    'avg_word_length', 'whitespace_ratio', 'alnum_special_ratio', 
    'digit_proportion', 'uppercase_proportion', 'max_word_length', 
    'min_word_length', 'code_markup_count', 'url_email_count', 
    'social_media_count', 'punctuation_proportion', 'stopword_proportion', 
    'language_count'
]

NONLINEAR_SELECTED_FEATURES = [ 
    'char_count',
    'word_count',
    'avg_word_length',
    'max_word_length',
    'code_markup_count',
    'url_email_count',
    'punctuation_proportion'
]

LINEAR_BOOST_SELECTED_FEATURES = NONLINEAR_SELECTED_FEATURES

MLP_PARAM_GRID = {
    'model__hidden_layer_sizes': [(100,), (250,), (100, 50), (150, 75, 25), (100, 100, 100)],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [0.0001, 0.001],
}
RF_PARAM_GRID = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [10, 20, None],
    'model__min_samples_leaf': [5, 10],
}
ET_PARAM_GRID = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [10, 20, None],
    'model__min_samples_leaf': [5, 10],
}

LINEAR_BOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}