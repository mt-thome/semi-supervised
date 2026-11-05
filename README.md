# Aprendizado Semissupervisionado com Digits

Implementação completa do plano experimental comparando **Baseline Supervisionado (`SVC`)**, **Self-Training (`SelfTrainingClassifier`)** e **Label Spreading (`LabelSpreading`)** (além de `LabelPropagation` e do Teto Supervisionado com 100% dos rótulos) no dataset `sklearn.datasets.load_digits`.

## Estrutura do Repositório

```text
atividade-semissupervisionado/
├── notebook.ipynb          # Experimento completo executado com análises, tabelas e gráficos
├── src/
│   ├── __init__.py
│   ├── data.py             # Carga do Digits, split 60/10/30 estratificado sem vazamento e máscara (-1)
│   ├── models.py           # Fábricas dos modelos (SVC, Self-Training, Label Spreading, Label Propagation)
│   ├── experiment.py       # Busca em grade na validação, loop 4 frações x 10 sementes e teste de Wilcoxon
│   ├── plots.py            # Funções de geração das 8 figuras do relatório
│   └── build_notebook.py   # Gerador reprodutível do notebook.ipynb
├── results/                # CSVs de resultados e figuras em alta resolução (.png)
├── relatorio.md            # Relatório técnico completo com fundamentação, resultados e discussão
└── README.md               # Instruções de uso e reprodutibilidade
```

## Ambiente e Reprodutibilidade

Para instalar as dependências e abrir o notebook:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter nbformat nbclient
jupyter notebook notebook.ipynb
```

Para re-executar o notebook via linha de comando:

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

