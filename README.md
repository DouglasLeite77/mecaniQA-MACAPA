# MecaniQA — OATs 1, 2 e 3

Projeto da equipe Macapá, da disciplina de Modelos de Aprendizagem de Máquina.
O objetivo é prever as trocas de óleo do dia seguinte a partir do histórico
diário de uma oficina, apoiando o planejamento de peças e equipe.

## Execução

Ambiente verificado: **Python 3.14**. Na pasta do projeto:

~~~powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 oat3_completa.py
~~~

O experimento completo grava tabelas, previsões, gráfico e conclusão em
[output/oat3](output/oat3). No VS Code, selecione o mesmo Python como kernel.

Para usar o Colab, abra [oat3_completa.ipynb](oat3_completa.ipynb), execute as
células na ordem e envie a planilha quando solicitado. O notebook inclui o
código necessário; não precisa dos scripts locais.

## Arquivos principais

| Etapa | Arquivo | Conteúdo |
| --- | --- | --- |
| OAT 1 | [mecaniqa_oat1.ipynb](mecaniqa_oat1.ipynb) | Inspeção, limpeza, IQR, decomposição e baselines de 7 e 30 dias |
| OAT 1 | [app.py](app.py) | Decomposição aditiva com período semanal |
| OAT 2 | [metricas_timeseries.ipynb](metricas_timeseries.ipynb) | Baselines, cinco folds temporais, métricas e atributos históricos |
| OAT 2 | [baseline_metricas.py](baseline_metricas.py) | Funções de baseline, lags e janelas rolantes |
| OAT 3 | [oat3_completa.ipynb](oat3_completa.ipynb) | Entrega completa dos quatro encontros |
| OAT 3 | [oat3_completa.py](oat3_completa.py) | Comparação inicial, Grid, Random e avaliação final |
| OAT 3 | [pipeline.py](pipeline.py) | Leitura validada, preparação, métricas e Grid de 07/10 |
| OAT 3 | [oat3_gridsearch.ipynb](oat3_gridsearch.ipynb) | Recorte do encontro de 07/10 |
| Apoio | [docs/GUIA_OAT3.md](docs/GUIA_OAT3.md) | Decisões, resultados e roteiro de apresentação |
| QA | [tests](tests) | Testes de dados, métricas, validação e resultados |
| Manutenção | [scripts](scripts) | Sincronização e execução dos notebooks |

Os nomes oat3_fechamento.ipynb e OAT_3_Tunando_Modelo_MecaniQA.ipynb continuam
disponíveis para os links já compartilhados. Ambos executam o mesmo
experimento completo. baseline_metricas.ipynb apresenta as mesmas funções
da OAT 2. A referência principal da entrega final é oat3_completa.ipynb.

## Dados e tratamento

A planilha [mecaniqa_dataset.xlsx](datasets/mecaniqa_dataset.xlsx) tem 731 dias,
de 01/01/2024 a 31/12/2025, e as colunas Data, Trocas_Oleo e Manutencao_Motor.
Há seis ausências em trocas de óleo, três em manutenção e cinco contagens
negativas em manutenção. Valores negativos viram ausentes; a planilha
original permanece intacta. A procedência detalhada da base ainda precisa
ser documentada pela equipe; usamos o arquivo fornecido na atividade.

A interpolação temporal e o IQR da OAT 1 servem à análise exploratória.
No experimento preditivo da OAT 3, a mediana e a escala são aprendidas dentro
do Pipeline, somente no treino de cada fold. Alvos ausentes são removidos.

## Protocolo da OAT 3

- Alvo: trocas de óleo de amanhã, usando as contagens observadas até hoje.
- 724 pares válidos: 579 para treino e 145 para teste, sem embaralhamento.
- Escolha entre Random Forest e Gradient Boosting pelo RMSE médio em cinco
  folds de TimeSeriesSplit(gap=1), dentro do treino.
- Grid e Random usam o mesmo espaço de parâmetros, RMSE e n_jobs=1.
  Grid avalia nove combinações para Random Forest; Random avalia seis.
- O teste final não escolhe modelo ou parâmetros. Empates de validação
  entre as buscas mantêm o Grid.
- Os dias previstos no teste vão de **09/08/2025 a 31/12/2025**. A avaliação
  prevê um dia por vez e pode usar a observação do dia anterior no próprio teste.

Na OAT 2, o código cria lag_1, lag_7, rolling_mean_7 e rolling_std_7, sempre
com dados anteriores ao alvo, e mostra os recortes temporais.
A OAT 3 mantém suas duas entradas originais para comparar o efeito da
sintonia sem mudar o experimento. A janela de sete dias é um baseline,
não uma entrada adicional dos modelos da OAT 3.

## Resultados verificados

Mesmas 145 datas para todos os modelos; valores arredondados:

| Modelo | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Naive | 6,42 | 8,85 | 26,64 |
| Média móvel de 7 dias | 7,35 | 8,12 | 30,89 |
| Random Forest padrão | 7,02 | 8,72 | 29,77 |
| Gradient Boosting padrão | 6,27 | 7,90 | 26,57 |
| Random Forest ajustado | 6,33 | 7,79 | 26,86 |

O ajuste escolheu **50 árvores e profundidade 3**. Grid e Random encontraram
a mesma configuração e o mesmo RMSE. O RMSE caiu 10,70% frente ao Random
Forest padrão e 4,05% frente à média móvel; o MAE caiu 13,82% frente à média móvel.
O Gradient Boosting teve o menor MAE no teste, enquanto o Random Forest
ajustado teve o menor RMSE. A escolha prévia usou o RMSE de validação.

MAE e RMSE incluem todos os dias. O MAPE considera somente demanda positiva;
neste teste são 145 de 145 dias. Os tempos são medidos a cada execução e
ficam em [comparacao.csv](output/oat3/comparacao.csv).
As médias dos folds da OAT 2 usam outros períodos e não são comparação
direta com este teste final.

## Testes e manutenção

~~~powershell
py -3.14 -m pip install -r requirements-dev.txt
py -3.14 -m pytest
py -3.14 scripts/testar_notebooks.py
~~~

O primeiro teste verifica métricas, datas, tratamento de nulos, baselines,
ausência de uso de observações futuras, buscas e reprodução das previsões
salvas. O segundo executa os sete notebooks em pastas temporárias, com um
kernel novo por notebook, e salva a evidência local em output/testes/.

Após modificar pipeline.py, baseline_metricas.py ou oat3_completa.py,
sincronize as cópias autossuficientes e execute os testes novamente:

~~~powershell
py -3.14 scripts/atualizar_notebooks.py
~~~

A sincronização limpa as saídas antigas. Execute os notebooks antes de
apresentá-los. O notebook da OAT 1 tem sua análise própria e é mantido
separadamente.

## Versionamento e entrega

Versionamos código, testes, dependências, base, documentação, apresentações
da equipe e resultados finais de output/oat3/. O .gitignore exclui caches,
ambientes virtuais, credenciais, output/testes/ e as pastas locais
docs/materiais/ e docs/aula slides/, conforme a organização da equipe.

O professor pede a entrega na branch main e a apresentação no modelo da
disciplina, com o nome mecaniQA_oat3_macapa.pdf. A execução dos scripts
não faz commit, push nem envio ao Blackboard.

## Equipe Macapá

- Artur Maia Coqueiro
- Douglas Renan Santos
- Kayky Ribeiro Souza
- Lenilson Dias Soares
- Joab Nascimento Rodrigues

Unidade: Vitória da Conquista.
