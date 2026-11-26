# MecaniQA — Modelos de Aprendizado de Máquina

Projeto da **Equipe Macapá** na disciplina *Modelos de Aprendizado de Máquina*
(Sistemas de Informação — UNEX, Vitória da Conquista, 2026.2).

A MecâniQA é uma plataforma para oficinas mecânicas. O objetivo deste MVP é
**prever a quantidade de trocas de óleo do dia seguinte** a partir do histórico
diário da oficina, apoiando o planejamento de estoque (óleo e filtros) e da equipe.

## Equipe

- Artur Maia Coqueiro
- Douglas Renan Santos
- Joab Nascimento Rodrigues
- Kayky Ribeiro Souza
- Lenilson Dias Soares

## Estrutura do repositório

```text
mecaniQA-MACAPA/
├── apresentacoes/      # PDFs apresentados em cada OAT
├── datasets/           # mecaniqa_dataset.xlsx (731 dias, 2024–2025)
├── notebooks/          # notebooks de cada OAT (rodam também no Colab)
│   └── legado/         # nomes antigos, mantidos para links já compartilhados
├── output/             # resultados salvos (tabelas, previsões e gráficos)
│   ├── oat2/
│   └── oat3/
├── scripts/            # manutenção: sincronizar e executar os notebooks
├── src/                # código Python reutilizável
├── tests/              # testes automatizados (pytest)
├── requirements.txt
└── requirements-dev.txt
```

| Etapa | Notebook | Código | Apresentação |
| --- | --- | --- | --- |
| OAT 1 — Compreensão e baseline | [mecaniqa_oat1.ipynb](notebooks/mecaniqa_oat1.ipynb) | [decomposicao_oat1.py](src/decomposicao_oat1.py) | [PDF](apresentacoes/mecaniQA_oat1_macapa.pdf) |
| OAT 2 — Baselines e métricas | [metricas_timeseries.ipynb](notebooks/metricas_timeseries.ipynb) | [baseline_metricas.py](src/baseline_metricas.py) | [PDF](apresentacoes/mecaniQA_oat2_Macapá.pdf) |
| OAT 3 — Tunando o modelo | [oat3_completa.ipynb](notebooks/oat3_completa.ipynb) | [oat3_completa.py](src/oat3_completa.py) | [PDF](apresentacoes/mecaniQA_oat3_Macapá.pdf) · [versão revisada](apresentacoes/mecaniQA_oat3_Macapá-atualizada.pdf) |

[`src/pipeline.py`](src/pipeline.py) concentra a leitura validada da planilha,
o Pipeline (imputação, padronização e modelo) e as métricas MAE, RMSE e MAPE.

## Como executar

Python 3.12 ou superior. Na raiz do repositório:

```powershell
py -m pip install -r requirements.txt
py src/oat3_completa.py          # experimento completo da OAT 3 (grava em output/oat3)
py src/baseline_metricas.py      # baselines e atributos da OAT 2
py src/decomposicao_oat1.py      # decomposição sazonal da OAT 1
```

Os notebooks podem ser abertos no VS Code/Jupyter a partir de `notebooks/` ou no
Google Colab (o notebook pede o envio da planilha quando não a encontra).

## Dados e tratamento

A planilha [mecaniqa_dataset.xlsx](datasets/mecaniqa_dataset.xlsx) tem 731 dias,
de 01/01/2024 a 31/12/2025, com as colunas `Data`, `Trocas_Oleo` e `Manutencao_Motor`.
Há 6 ausências em trocas de óleo, 3 em manutenção e 5 contagens negativas em
manutenção (14 registros tratados). Valores negativos viram ausentes; a planilha
original permanece intacta.

A interpolação temporal e o IQR da OAT 1 servem à análise exploratória. No
experimento preditivo, mediana e escala são aprendidas dentro do Pipeline,
somente no treino de cada fold, e alvos ausentes são removidos.

## Protocolo de avaliação (OAT 3)

- **Alvo:** trocas de óleo de amanhã, usando as contagens observadas até hoje.
- **Separação temporal:** 724 pares válidos — 579 para treino e 145 para teste
  (09/08/2025 a 31/12/2025), sem embaralhamento.
- **Seleção:** Random Forest × Gradient Boosting pelo RMSE médio em 5 folds de
  `TimeSeriesSplit(gap=1)`, apenas no treino.
- **Sintonia:** Grid Search (9 combinações) e Random Search (6) no mesmo espaço
  de parâmetros; o teste final só é consultado depois das escolhas.

Na OAT 2 são criados `lag_1`, `lag_7`, `rolling_mean_7` e `rolling_std_7`, sempre
com dados anteriores ao alvo. O modelo da OAT 3 mantém as duas entradas originais
para isolar o efeito da sintonia de hiperparâmetros.

## Resultados

### OAT 2 — média dos 5 folds temporais

| Baseline | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Naive | 6,65 | 8,92 | 32,64 |
| Média móvel de 7 dias | 7,36 | 8,09 | 37,22 |

Detalhes por fold e previsões em [output/oat2](output/oat2).

### OAT 3 — mesmas 145 datas de teste

| Modelo | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Naive | 6,42 | 8,85 | 26,64 |
| Média móvel de 7 dias | 7,35 | 8,12 | 30,89 |
| Random Forest padrão | 7,02 | 8,72 | 29,77 |
| Gradient Boosting padrão | 6,27 | 7,90 | 26,57 |
| **Random Forest ajustado** (50 árvores, profundidade 3) | 6,33 | **7,79** | 26,86 |

O ajuste reduziu o RMSE em 10,70% frente ao Random Forest padrão. O Gradient
Boosting teve o menor MAE no teste; a escolha prévia usou o RMSE de validação.
Tabelas, previsões, tempos das buscas e gráfico em [output/oat3](output/oat3).

**Limitações e próximos passos:** a série tem sazonalidade semanal forte
(sexta a domingo com ~75% mais trocas). Incluir o dia da semana e o `lag_7` como
entradas do modelo e comparar com um Naive sazonal são os próximos passos naturais.

## Testes

```powershell
py -m pip install -r requirements-dev.txt
py -m pytest                          # testes de dados, métricas, validação e resultados
py scripts/testar_notebooks.py        # executa todos os notebooks em pastas temporárias
```

Ao alterar algo em `src/`, sincronize os notebooks e grave as novas saídas:

```powershell
py scripts/atualizar_notebooks.py
py scripts/testar_notebooks.py --salvar
```
