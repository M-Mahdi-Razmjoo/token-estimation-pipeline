BEST_HYPERPARAMS = {
    'rf': {
        'r50k': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 50, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'cl100k': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 50, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'o200k': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 50, 'model__max_depth': 20, 'model__min_samples_leaf': 5}
        },
        'deepseek_r1': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'qwen_qwq': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'llama3_1_8b': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 50, 'model__max_depth': 20, 'model__min_samples_leaf': 5}
        }
    },
    'et': {
        'r50k': {
            'all': {'model__n_estimators': 50, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'cl100k': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'o200k': {
            'all': {'model__n_estimators': 100, 'model__max_depth': 20, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 50, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'deepseek_r1': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'qwen_qwq': {
            'all': {'model__n_estimators': 100, 'model__max_depth': None, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 50, 'model__max_depth': None, 'model__min_samples_leaf': 5}
        },
        'llama3_1_8b': {
            'all': {'model__n_estimators': 100, 'model__max_depth': 20, 'model__min_samples_leaf': 5},
            'english': {'model__n_estimators': 100, 'model__max_depth': 20, 'model__min_samples_leaf': 5}
        }
    },
    'mlp': {
        'r50k': {
            'all': {'model__activation': 'relu', 'model__alpha': 0.0001, 'model__hidden_layer_sizes': (100, 100, 100)},
            'english': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (100, 100, 100)}
        },
        'cl100k': {
            'all': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (150, 75, 25)},
            'english': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (150, 75, 25)}
        },
        'o200k': {
            'all': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (100, 100, 100)},
            'english': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (150, 75, 25)}
        },
        'deepseek_r1': {
            'all': {'model__activation': 'relu', 'model__alpha': 0.0001, 'model__hidden_layer_sizes': (100, 100, 100)},
            'english': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (100, 100, 100)}
        },
        'qwen_qwq': {
            'all': {'model__activation': 'relu', 'model__alpha': 0.0001, 'model__hidden_layer_sizes': (100, 100, 100)},
            'english': {'model__activation': 'relu', 'model__alpha': 0.0001, 'model__hidden_layer_sizes': (100, 100, 100)}
        },
        'llama3_1_8b': {
            'all': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (100, 100, 100)},
            'english': {'model__activation': 'relu', 'model__alpha': 0.001, 'model__hidden_layer_sizes': (150, 75, 25)}
        }
    },
    'linear': {
        'r50k': {'all': {}, 'english': {}},
        'cl100k': {'all': {}, 'english': {}},
        'o200k': {'all': {}, 'english': {}},
        'deepseek_r1': {'all': {}, 'english': {}},
        'qwen_qwq': {'all': {}, 'english': {}},
        'llama3_1_8b': {'all': {}, 'english': {}},
    }
}