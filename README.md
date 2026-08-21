# MecaniQA - OAT 1: Compreensão e Baseline

Este repositório faz parte da primeira OAT da disciplina de Modelos de Aprendizagem de Máquina. O trabalho está sendo desenvolvido pela equipe Macapá, da unidade de Vitória da Conquista.

## Sobre o projeto

A proposta da MecaniQA é analisar o histórico de manutenções de oficinas e auto centers. A ideia é entender os períodos de maior e menor movimento para ajudar no planejamento do estoque de peças e da equipe de mecânicos.

Nesta primeira etapa, estamos trabalhando com séries temporais de trocas de óleo e manutenções de motor. O projeto envolve a organização e limpeza dos dados, a análise de tendência e sazonalidade e a criação de modelos simples de previsão para servir como base de comparação.

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

O código atual realiza a leitura da planilha, transforma a coluna de data em índice e ordena os registros em ordem cronológica. Também faz o preenchimento dos valores ausentes da série analisada e gera a decomposição visual das trocas de óleo.

A equipe escolheu o modelo aditivo e o período de 7 dias porque os dados são diários e apresentam um comportamento que se repete semanalmente. O resultado é apresentado em quatro gráficos: série observada, tendência, sazonalidade e ruído.

As etapas de tratamento de outliers e dos modelos Naive e de médias móveis ainda serão acrescentadas para completar a OAT 1.

## Estrutura do repositório

```text
mecaniQA-MACAPA/
|-- app.py
|-- datasets/
|   `-- mecaniqa_dataset.xlsx
`-- README.md
```

## Como executar

Com o Python instalado, abra o terminal na pasta do projeto e instale as bibliotecas utilizadas:

```bash
pip install pandas matplotlib statsmodels openpyxl
```

Depois execute:

```bash
python app.py
```

O programa carregará o dataset e exibirá os gráficos da decomposição da série temporal.

## Equipe Macapá

- Artur Maia Coqueiro
- Douglas Renan Santos
- Kayky Ribeiro Souza
- Lenilson Dias Soares
- Joab Nascimento Rodrigues

**Unidade:** Vitória da Conquista
