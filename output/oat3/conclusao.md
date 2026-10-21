# MecaniQA - conclusão da OAT 3

O modelo escolhido na validação foi **Random Forest**. As duas buscas utilizaram o mesmo espaço de parâmetros, cinco folds de TimeSeriesSplit com gap=1, o mesmo RMSE e uma única tarefa de processamento. Os tempos abaixo medem somente o fit de cada busca, incluindo o refit.

- Grid Search: 9 combinações, 4.65 segundos, RMSE de validação 10.1835 e RMSE de teste 7.7908.
- Random Search: 6 combinações, 3.34 segundos, RMSE de validação 10.1835 e RMSE de teste 7.7908.

O Random Search levou 1.31 segundos a menos. As duas buscas tiveram o mesmo RMSE no teste. A escolha final foi **Grid Search**, pelo RMSE na validação, antes de consultar o teste; em caso de empate, mantemos o Grid conforme a regra definida no experimento.

Random Search é uma opção quando o espaço de parâmetros é grande e há limite de processamento. Nesta base pequena, a diferença de tempo é uma medição desta execução; é necessário observar conjuntamente o custo e o erro, sem garantir que uma busca será sempre mais rápida ou mais precisa.

O RMSE final mudou 10.70% em relação ao modelo inicial padrão (percentual positivo indica redução; negativo indica piora). O Grid reduziu o RMSE em 10.72% frente ao Random Forest anterior de 200 árvores, reexecutado nas mesmas condições.

Para explicar o resultado ao dono da oficina, o MAE é a medida mais direta: a média móvel errou 7.35 trocas de óleo por dia, enquanto o modelo ajustado errou 6.33. O RMSE destaca erros maiores. O MAPE considera somente dias com demanda positiva e deve ser lido junto das outras métricas. Esses erros são quantidades de serviços; não medem lucro em reais sem custos de falta e excesso de estoque.

O teste final contém 145 dias. Esta avaliação prevê um dia por vez usando observações disponíveis até o dia anterior à previsão; não é uma previsão de todo o período de teste de uma única vez.
