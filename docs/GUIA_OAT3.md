# MecaniQA — guia da OAT 3

A apresentação usa o modelo do professor e cabe em até sete minutos. O texto de apoio abaixo fica separado dos slides.

- [Apresentação editável](https://docs.google.com/presentation/d/1x2fY3TadtFeb3VEib0HIdJrWsIWzMOdvqymlRgXB08g/edit)
- [PDF para entrega](mecaniQA_oat3_macapa.pdf)
- [Dados dos gráficos](https://docs.google.com/spreadsheets/d/1qi2E5P4IXiCqAx_sfSF69FkIyXfntueI2SWUjdbpaCc/edit)
- [Notebook completo](../oat3_completa.ipynb)
- [Resultados e gráfico](../output/oat3)

## Execução

~~~powershell
py -3.14 -m pip install -r requirements-dev.txt
py -3.14 -m pytest
py -3.14 scripts/testar_notebooks.py
py -3.14 oat3_completa.py
~~~

Os testes de notebooks usam pastas temporárias e guardam a evidência local em output/testes, ignorado pelo Git. O experimento do último comando atualiza os resultados finais; ao executá-lo novamente, os tempos podem mudar.

## Roteiro de fala

### 1. Abertura

Tempo sugerido: 20 s. Apresentar a equipe e o objetivo: prever a quantidade de trocas de óleo de amanhã usando observações disponíveis até hoje. Esta etapa compara modelos de regressão e a sintonia de hiperparâmetros.
Fonte: repositório da equipe Macapá; enunciado OAT 3.

### 2. Agenda

Tempo sugerido: 20 s. A apresentação acompanha quatro encontros: modelos iniciais (30/09), Grid Search (07/10), Random Search (14/10) e comparação final (21/10). Destacar os resultados e usar o notebook para detalhes de código.
Fonte: CADERNO DE GUIAS - OAT 3 - TUTORIAIS.preenchido.docx.pdf.

### 3. Validação temporal

Tempo sugerido: 55 s. A base possui 731 dias. Seis valores ausentes de trocas de óleo retiram seis alvos, e o último dia não tem alvo do dia seguinte; restam 724 pares. São 579 exemplos de treino e 145 de teste. As contagens negativas de manutenção viram ausentes. Mediana e escala são aprendidas em cada treino. TimeSeriesSplit(gap=1) protege a fronteira do alvo em t+1. O teste cobre previsões de 09/08/2025 a 31/12/2025. Trata-se de avaliação diária, podendo observar dias anteriores do teste.
A OAT 2 contém lags e janelas; a OAT 3 mantém Trocas_Oleo e Manutencao_Motor como entradas para isolar a comparação dos hiperparâmetros.
Fontes: datasets/mecaniqa_dataset.xlsx; pipeline.py; oat3_completa.py.

### 4. Modelos iniciais

Tempo sugerido: 60 s. Random Forest e Gradient Boosting usam hiperparâmetros padrão de aprendizado e semente 42. O Random Forest foi escolhido no treino por RMSE médio de validação de 11,9982, contra 12,4709 do Gradient Boosting. A tabela mostra o teste posterior à escolha. O Gradient Boosting tem o menor MAE e MAPE no teste; isso não muda a escolha feita previamente pela validação. O RF padrão não supera a média móvel em RMSE. MAE e RMSE usam todos os dias; o MAPE considera somente demanda positiva, que corresponde aos 145 dias neste teste.
Fonte: output/oat3/metricas.csv e saída de comparar_modelos.

### 5. Grid e Random Search

Tempo sugerido: 65 s. Grid: n_estimators=[50,100,200] e max_depth=[3,5,10], nove combinações. Random usa o mesmo espaço, n_iter=6 e semente 42. Ambos usam cinco folds temporais, neg_root_mean_squared_error e n_jobs=1. Os tempos incluem fit e refit: 4,52 s e 3,68 s. A diferença foi 0,84 s. Ambos encontraram profundidade 3 e 50 árvores; RMSE no teste 7,7908 para ambos, diferença de erro zero. O empate de validação mantém o Grid pela regra do experimento. Uma execução não prova que Random será sempre mais rápido.
Fonte: output/oat3/comparacao.csv; oat3_completa.py.

### 6. Previsões no teste

Tempo sugerido: 65 s. O gráfico compara valores reais, média móvel de sete dias e Random Forest ajustado nos últimos 60 dias do teste. As métricas foram calculadas nos 145 dias, não apenas neste recorte. O ajustado ainda erra em picos e quedas; o objetivo é reduzir o erro agregado. As previsões são de um dia à frente e não de todo o horizonte de 145 dias em uma única chamada operacional.
Fonte: output/oat3/previsoes.csv. Dados e fórmulas do gráfico: https://docs.google.com/spreadsheets/d/1qi2E5P4IXiCqAx_sfSF69FkIyXfntueI2SWUjdbpaCc/edit

### 7. Resultado final

Tempo sugerido: 65 s. A média móvel teve MAE 7,3478 e o RF ajustado 6,3321: redução de 13,82%. Em RMSE, o ajustado caiu 10,70% frente ao RF padrão e 4,05% frente à média móvel. O percentual de 10,72% que aparece no guia usa outra referência: o RF anterior de 200 árvores, reexecutado nas mesmas condições. Não misturar essas bases. MAE explica o erro médio em serviços por dia. Não foram calculados custos de estoque ou lucro em reais. O Gradient Boosting manteve MAE ligeiramente menor no teste; a seleção usou RMSE na validação.
Fonte: output/oat3/metricas.csv; cenarios.csv; conclusao.md.

### 8. Fontes

Tempo sugerido: 15 s. Indicar os arquivos para reprodução. A procedência detalhada do dataset não está documentada; utilizamos a planilha fornecida na atividade. Os materiais do professor são referências locais. Tempo total sugerido: 6 min 5 s.
Fontes: datasets/mecaniqa_dataset.xlsx; oat3_completa.ipynb; output/oat3; enunciado OAT 3 e caderno de guias da disciplina.

## Como explicar as métricas

MAE mede o erro médio em trocas de óleo por dia. RMSE dá mais peso aos erros grandes. MAPE considera só demanda positiva; são 145 de 145 dias no teste final. Não compare diretamente as médias dos folds da OAT 2 com este teste da OAT 3.

Os três percentuais têm referências diferentes:

| Comparação | Redução do RMSE |
| --- | ---: |
| RF ajustado × RF padrão (100 árvores) | 10,70% |
| RF ajustado × RF anterior (200 árvores) | 10,72% |
| RF ajustado × média móvel de 7 dias | 4,05% |

A redução de **13,82%** usa o **MAE** contra a média móvel de sete dias. O Gradient Boosting teve MAE e MAPE menores no teste; o RF ajustado teve o menor RMSE. A escolha do modelo usou o RMSE de validação do treino.

## Entrega

O nome do PDF é mecaniQA_oat3_macapa.pdf. O professor pede o arquivo na branch main e no formulário de apresentação do Blackboard. Salvar ou executar o projeto não publica a entrega. Os materiais do professor ficam disponíveis localmente em docs/materiais e docs/aula slides e são ignorados no Git.
