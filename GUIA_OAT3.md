# MecaniQA - roteiro do piloto para a OAT 3

## Arquivos e execução

| Uso | Arquivo |
| --- | --- |
| Apresentar somente 07/10 | `oat3_gridsearch.ipynb` |
| Apresentar todos os encontros | `oat3_completa.ipynb` |
| Executar somente o Grid pelo terminal | `py -3.14 pipeline.py` |
| Executar todas as etapas pelo terminal | `py -3.14 oat3_completa.py` |
| Dados | `datasets/mecaniqa_dataset.xlsx` |
| Tabelas, gráfico e conclusão da execução completa | `output/oat3` |

No Colab, faça upload do notebook escolhido, conecte o ambiente e execute as células de cima para baixo. Quando solicitado, envie a planilha. Os notebooks contêm o código completo e não precisam de outros scripts. Salve uma cópia no Drive e mantenha as saídas visíveis para apresentar.

## 30/09 - modelos iniciais

**Escolha dos algoritmos:** Random Forest e Gradient Boosting são modelos de regressão que capturam relações não lineares e interações entre as variáveis. Comparamos os dois com os hiperparâmetros de aprendizado padrão, fixando somente a semente aleatória em 42 para repetir o experimento. A sazonalidade não é garantida apenas pelo algoritmo; depende das informações usadas como entradas.

**Linha do tempo:** organizamos os registros por data e separamos os primeiros 80% dos exemplos com alvo válido para treino e os 20% finais para teste. Não embaralhamos os dados. A imputação e a escala são aprendidas apenas no treino de cada fold.

O piloto executa a comparação inicial e mostra o RMSE e o MAPE dos modelos e dos baselines na tabela final. Mostre que todas as linhas da tabela usam as mesmas datas de teste.

## 07/10 - Grid Search

**Espaço de busca:** o Random Forest foi escolhido pelo menor RMSE médio na validação dos dados de treino. O grid testa `n_estimators=[50, 100, 200]` e `max_depth=[3, 5, 10]`, totalizando nove combinações. Esses valores limitam o custo da busca e ajudam a controlar a complexidade. A profundidade limitada reduz o risco de overfitting, sem garantir sua ausência.

**Proteção temporal:** o K-Fold comum pode treinar com períodos posteriores para validar períodos anteriores, mesmo sem embaralhamento. Por isso usamos `TimeSeriesSplit(n_splits=5, gap=1)` dentro do `GridSearchCV`. O intervalo de uma linha protege a disponibilidade do alvo do dia seguinte. O teste final não participa da escolha do modelo nem dos parâmetros.

**Fala do piloto:** "O código testou nove combinações em cinco divisões temporais. A melhor configuração foi 50 árvores e profundidade 3. No teste separado, o RMSE passou de 8,7263 para 7,7908, uma redução de 10,72% em relação à referência anterior de 200 árvores."

Esses valores foram medidos neste checkout. Após mudar dados ou parâmetros, execute novamente e use os resultados da nova execução.

## 14/10 - Random Search

**Trade-off:** Random Search é uma opção quando o espaço de parâmetros é grande e existe limite de tempo ou processamento. Em vez de testar todas as combinações, avaliamos uma amostra e verificamos se a diferença de erro compensa a economia de tempo.

O piloto e o copiloto executam `RandomizedSearchCV` com `n_iter=6`. As duas buscas usam o mesmo grid, os mesmos folds, o mesmo RMSE e `n_jobs=1`. O tempo de cada `fit`, incluindo o refit, é medido com `perf_counter`.

Mostre `resultado['comparacao']`. O QA confere a comparação e auxilia o Arquiteto a revisar o bloco Markdown gerado. Use os tempos desta execução, pois eles variam conforme a máquina.

Na execução verificada, as duas buscas encontraram a mesma configuração e tiveram o mesmo RMSE. Não diga que Random Search é sempre mais rápido ou mais preciso.

## 21/10 - fechamento

**Métrica para explicar o ganho:** o MAE expressa o erro médio em trocas de óleo por dia e costuma ser o mais direto para o dono da oficina. O RMSE destaca erros maiores. O MAPE exige cuidado nos dias de demanda zero. Não convertemos essas medidas em lucro em reais sem conhecer os custos de falta e excesso.

Mostre a tabela dos três cenários (`resultado['cenarios']`): média móvel de 7 dias, modelo inicial padrão e modelo ajustado. Depois mostre o gráfico sobreposto com os valores reais, o baseline, o modelo inicial e o ajustado. O gráfico usa os últimos 60 dias do teste para facilitar a leitura.

Finalize mostrando a conclusão gerada com os tempos, erros e diferenças medidos. O modelo final é escolhido pelo RMSE na validação; em caso de empate entre as buscas, o experimento mantém o Grid.

## Como interpretar os percentuais

- **10,72%:** redução do RMSE do Grid em relação ao Random Forest anterior de 200 árvores, reexecutado nas mesmas condições.
- **10,70%:** redução do RMSE em relação ao Random Forest com parâmetros padrão (100 árvores nesta versão da biblioteca).
- Fórmula: `(erro inicial - erro ajustado) / erro inicial * 100`.
- Valor negativo representa piora, e erro inicial zero torna o percentual indefinido.

O MAPE usa somente dias com demanda positiva. Neste teste, os 145 dias têm demanda positiva; a contagem é impressa. A previsão é feita um dia por vez, podendo usar observações dos dias anteriores do teste. Isso difere de prever todos os 145 dias de uma única vez.

## Conferência antes da entrega

Revise com a equipe as respostas e os resultados, execute todas as células e confirme que nenhuma terminou com erro. O PDF do professor pede que a entrega final esteja na branch `main`. O código não realiza commit nem push; a publicação dos arquivos revisados é um passo separado.
