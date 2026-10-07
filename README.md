# MecaniQA - OAT 1: Compreensão e Baseline

Este repositório faz parte da primeira OAT da disciplina de Modelos de Aprendizagem de Máquina. O trabalho está sendo desenvolvido pela equipe Macapá, da unidade de Vitória da Conquista.

## Sobre o projeto

A proposta da MecaniQA é analisar o histórico de manutenções de oficinas e auto centers. A ideia é entender os períodos de maior e menor movimento para ajudar no planejamento do estoque de peças e da equipe de mecânicos.

Nesta primeira etapa, trabalhamos com séries temporais de trocas de óleo e manutenções de motor. O projeto envolve a organização e limpeza dos dados, a análise de tendência e sazonalidade e a criação de modelos simples de previsão para servir como base de comparação.

## Objetivos da OAT 1

- Organizar os registros usando a data como índice.
- Identificar e tratar valores ausentes e valores fora do padrão.
- Visualizar a série completa ao longo do tempo.
- Separar a série em tendência, sazonalidade e ruído.
- Criar uma previsão Naive, usando o valor do dia anterior.
- Criar previsões com médias móveis de 7 e 30 dias.
- Comparar os dados reais com as previsões geradas.

## Base de dados

O arquivo utilizado está na pasta `datasets` e possui 731 registros, entre janeiro de 2024 e dezembro de 2025. As colunas principais são:

- `Data`: dia em que o registro foi realizado.
- `Trocas_Oleo`: quantidade de trocas de óleo.
- `Manutencao_Motor`: quantidade de manutenções de motor.

## Etapa atual

O arquivo `app.py` realiza a leitura da planilha, transforma a coluna de data em índice e ordena os registros em ordem cronológica. Também faz o preenchimento dos valores ausentes da série analisada e gera a decomposição visual das trocas de óleo.

A equipe escolheu o modelo aditivo e o período de 7 dias porque os dados são diários e apresentam um comportamento que se repete semanalmente. O resultado é apresentado em quatro gráficos: série observada, tendência, sazonalidade e ruído.

O arquivo `pipeline.py` prevê as trocas de óleo do dia seguinte e implementa a atividade da OAT 3 de 07/10/2026. Após remover os alvos ausentes, os 724 exemplos válidos são separados cronologicamente em 579 para treino e 145 para teste. Random Forest e Gradient Boosting são comparados por RMSE médio em cinco folds de `TimeSeriesSplit`, somente dentro do treino. O modelo escolhido é ajustado com `GridSearchCV`; os melhores parâmetros e a redução percentual do RMSE são impressos ao final.

A imputação por mediana e a padronização ficam dentro do Pipeline e são aprendidas em cada treino. Quantidades negativas são consideradas ausentes, sem alterar a planilha. O `TimeSeriesSplit` usa `gap=1`, pois o alvo de cada linha é do dia seguinte. A comparação final inclui Naive, média móvel de 7 dias, os dois modelos iniciais e o modelo ajustado, todos nas mesmas datas. As previsões são diárias: a demanda observada de hoje pode ser usada para prever amanhã.

O RMSE e o MAE usam todos os dias do teste. Como existe demanda zero na base, o MAPE é informado somente para os dias com demanda maior que zero, com a quantidade de dias incluídos. O resultado histórico de MAE 7,04 não é usado para calcular ganhos: o modelo inicial é treinado novamente com o mesmo tratamento e o mesmo teste do modelo ajustado.

O notebook `oat3_gridsearch.ipynb` executa essa rotina e apresenta a tabela de métricas, os resultados da busca e as previsões para a atividade do dia 7.

O arquivo `mecaniqa_oat1.ipynb` reúne a entrega completa em formato Jupyter Notebook: inspeção com `head()` e `info()`, tratamento de valores ausentes e outliers pelo método IQR, decomposição sazonal aditiva, Pipeline sem vazamento de dados, modelos Naive e médias móveis de 7 e 30 dias e gráfico comparando valores reais e previsões.

## Estrutura do repositório

```text
mecaniQA-MACAPA/
|-- app.py
|-- pipeline.py
|-- oat3_gridsearch.ipynb
|-- mecaniqa_oat1.ipynb
|-- datasets/
|   `-- mecaniqa_dataset.xlsx
|-- mecaniQA_oat1_macapa.pdf
`-- README.md
```

## Como executar

Com o Python instalado, abra o terminal na pasta do projeto e instale as bibliotecas utilizadas:

```bash
py -3.14 -m pip install pandas matplotlib statsmodels openpyxl scikit-learn
```

Depois execute:

```bash
py -3.14 app.py
```

O programa carregará o dataset e exibirá os gráficos da decomposição da série temporal.

Para treinar e avaliar o Pipeline preditivo, execute:

```bash
py -3.14 pipeline.py
```

Para abrir a entrega completa em formato Notebook, abra `mecaniqa_oat1.ipynb` no VS Code com a extensão Jupyter ou no Google Colab.

### OAT 3 no Colab

Abra `oat3_gridsearch.ipynb` no Google Colab pela opção de upload de notebook e execute todas as células. Quando solicitado, envie `datasets/mecaniqa_dataset.xlsx` do computador. O notebook contém o código completo de treinamento; não é necessário enviar `pipeline.py`. Localmente, ele usa a planilha da pasta `datasets` ou uma cópia ao lado do notebook.

## Equipe Macapá

- Artur Maia Coqueiro
- Douglas Renan Santos
- Kayky Ribeiro Souza
- Lenilson Dias Soares
- Joab Nascimento Rodrigues

**Unidade:** Vitória da Conquista
