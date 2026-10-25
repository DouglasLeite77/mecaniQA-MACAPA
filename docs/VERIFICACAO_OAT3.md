# Verificação da OAT 3

Data: 25/10/2026. Ambiente: Python 3.14.3.

## Resultado

- 38 testes automatizados aprovados.
- Sete notebooks executados do início ao fim em kernels novos.
- Quatro scripts executados sem falha.
- Previsões e métricas do experimento completo reproduzidas.
- Apresentação de oito slides exportada no nome exigido.

A evidência resumida está em [verificacao.json](../output/oat3/verificacao.json).
Logs, notebooks de teste e imagens de inspeção ficam em output/testes,
ignorado pelo Git.

## Requisitos do professor

| Encontro | Verificação | Evidência |
| --- | --- | --- |
| 30/09 | Random Forest e Gradient Boosting com parâmetros padrão, treino temporal, RMSE e MAPE | oat3_completa.ipynb; metricas.csv |
| 07/10 | Grid definido, GridSearchCV com TimeSeriesSplit, melhores parâmetros e ganho percentual | oat3_completa.ipynb; oat3_gridsearch.ipynb |
| 14/10 | RandomizedSearchCV com seis iterações, tempos comparáveis e diferença de erro | comparacao.csv; conclusao.md |
| 21/10 | MAE, RMSE e MAPE dos três cenários; gráfico real × baseline × ajustado | cenarios.csv; comparacao_final.png |
| Apresentação | Modelo da disciplina, autores, gráficos e nome de arquivo | mecaniQA_oat3_macapa.pdf |

Todos os caminhos de resultados da tabela pertencem a
[output/oat3](../output/oat3). A referência completa de execução é
[oat3_completa.ipynb](../oat3_completa.ipynb).

## Correções confirmadas pelos testes

- O notebook de métricas agora encontra a planilha na pasta datasets e calcula
  o MAPE somente nos dias com demanda positiva.
- As versões alternativas da OAT 3 usam o mesmo experimento completo.
  Não substituem arquivos ausentes por dados simulados.
- As comparações usam as mesmas datas válidas para todos os modelos.
- Importar os scripts exploratórios não abre gráficos nem executa leituras.
- Baselines e atributos históricos usam apenas dados disponíveis antes do alvo.
- Datas inválidas, duplicadas, com lacunas e métricas desalinhadas são rejeitadas.

## Interpretação

O RF ajustado teve MAE 6,3321 e RMSE 7,7908. O ganho de MAE foi 13,82%
contra a média móvel. O ganho de RMSE foi 10,70% contra o RF padrão.
Grid e Random encontraram a mesma configuração: 50 árvores e profundidade 3.
Os tempos registrados variam entre execuções.

O Gradient Boosting teve o menor MAE no teste. A escolha do algoritmo usou
o RMSE da validação de treino, sem consultar esse teste. Não foi estimado
lucro em reais.

## Entrega externa

O PDF está em [mecaniQA_oat3_macapa.pdf](mecaniQA_oat3_macapa.pdf).
O versionamento na branch main e o envio no Blackboard precisam ser
conferidos separadamente; os testes não executam essas ações.
