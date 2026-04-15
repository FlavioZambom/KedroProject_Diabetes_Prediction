# Diabetes Prediction - Kedro Project

Prevendo a incidência de Diabetes usando pipelines Kedro.

## Instalação

```bash
# Instalar uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependências
uv sync
```

## Executar pipelines

```bash
# Rodar pipeline completo (data engineering + training)
uv run kedro run

# Rodar pipelines individualmente
uv run kedro run --pipeline data_engineering
uv run kedro run --pipeline training
uv run kedro run --pipeline inference
```

## Visualizar pipelines

```bash
uv run kedro viz
```

## Estrutura do projeto

```
├── conf/base/          # Configurações (catalog.yml, parameters.yml)
├── data/01_raw/        # Dados brutos (CSVs)
├── src/diabetes_prediction/
│   ├── pipelines/
│   │   ├── data_engineering/   # Limpeza, features, encoding, scaling
│   │   ├── training/           # Treinamento e avaliação do modelo
│   │   └── inference/          # Predição em novos dados
│   └── pipeline_registry.py
└── pyproject.toml
```

## Pipelines

### Data Engineering
`raw_data → clean → features → split → fit/transform encoders → fit/transform scaler → master_table`

### Training
`master_table → train_model → evaluate_model → metrics`

### Inference
`inference_data → clean → features → transform_encoders → transform_scaler → predict → predictions`
