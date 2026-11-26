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

Tempo planejado: **5 min 50 s**. Ensaie com cronômetro e ajuste as palavras ao seu jeito de falar. O roteiro é apoio; use o gráfico e a tabela para explicar os resultados.

### 1. Abertura — cerca de 20 s

Boa tarde, professor. Somos a equipe Macapá. Na OAT 3, comparamos dois modelos e ajustamos seus parâmetros para prever as trocas de óleo do dia seguinte. Vamos mostrar como fizemos a avaliação e qual foi o ganho em relação aos baselines.

### 2. Agenda — cerca de 15 s

Primeiro vamos explicar a separação dos dados. Depois mostramos os modelos iniciais, as duas buscas de parâmetros e a comparação final entre as previsões e os valores reais.

### 3. Validação temporal — cerca de 55 s

Usamos a mesma planilha das etapas anteriores, com 731 dias de histórico. Depois de retirar os pares sem alvo observado, ficaram 724 exemplos: 579 para treino e 145 para teste. Mantivemos a ordem das datas. A escolha dos modelos e dos parâmetros aconteceu somente no treino, com cinco divisões temporais. A imputação e a escala também foram ajustadas dentro de cada treino. O intervalo de uma linha entre treino e validação considera que nosso alvo é a demanda de amanhã. A previsão é diária: para prever amanhã, podemos usar as observações de hoje.

### 4. Modelos iniciais — cerca de 60 s

Comparamos Random Forest e Gradient Boosting com os parâmetros padrão e fixamos a semente em 42 para repetir o experimento. A escolha foi pelo menor RMSE médio na validação: 12,00 para o Random Forest e 12,47 para o Gradient Boosting. Por isso ajustamos o Random Forest. Esta tabela mostra o teste final, consultado depois da escolha. Nele, o Gradient Boosting teve o menor MAE e MAPE entre os modelos iniciais. Isso mostra que o resultado depende da métrica e do período avaliado.

### 5. Grid e Random Search — cerca de 65 s

No Grid Search, testamos 50, 100 e 200 árvores, com profundidades 3, 5 e 10, totalizando nove combinações. No Random Search, avaliamos seis combinações do mesmo espaço. Mantivemos os mesmos folds, a mesma métrica e o mesmo número de tarefas de processamento. As duas buscas encontraram 50 árvores e profundidade 3. O RMSE de validação e o de teste foram iguais entre as buscas. Nesta execução, o Grid levou 4,52 segundos e o Random, 3,68: uma diferença de 0,84 segundo. O empate na validação manteve o Grid, conforme a regra do experimento. Esses tempos podem variar em outra execução.

### 6. Previsões no teste — cerca de 55 s

A linha preta mostra a quantidade real de trocas de óleo. A cinza tracejada é a média móvel, e a verde é o Random Forest ajustado. Mostramos os últimos 60 dias para facilitar a leitura, mas as métricas usam todos os 145 dias do teste. A média móvel suaviza bastante as oscilações. O modelo ajustado varia mais, embora ainda erre em vários picos e quedas. Portanto, o resultado deve ser avaliado pelo conjunto das métricas, além do gráfico.

### 7. Resultado final — cerca de 65 s

Na comparação com a média móvel, o erro médio absoluto caiu de 7,35 para 6,33 trocas de óleo por dia. Essa redução foi de 13,82%. Em relação ao Random Forest inicial, o RMSE caiu 10,70%. O ajustado teve o menor RMSE entre os modelos avaliados; o Gradient Boosting manteve o menor MAE. Escolhemos o ajuste pelo RMSE da validação, antes de consultar o teste. Para a oficina, o MAE é mais fácil de explicar porque está na unidade do serviço. Esses números medem erro de previsão. Para calcular lucro ou economia em reais, precisaríamos conhecer os custos da operação.

### 8. Fontes — cerca de 15 s

Os resultados podem ser conferidos na planilha, no notebook completo e nos arquivos de saída do repositório. Usamos também o enunciado e o caderno de guias da disciplina. Obrigado.

## Perguntas que podem aparecer

**Por que escolheram Random Forest se o Gradient Boosting aparece melhor na tabela inicial?**

Porque a escolha usou o RMSE médio da validação no treino: 11,9982 para RF e 12,4709 para GB. A tabela inicial mostra o teste final, que só foi consultado depois da seleção. O GB ficou melhor em MAE e MAPE no teste; isso deve ser reconhecido.

**Por que usar TimeSeriesSplit?**

Para treinar com períodos anteriores e validar em períodos posteriores. O K-Fold comum pode treinar com dados posteriores à validação. O gap de uma linha considera o alvo deslocado para o dia seguinte.

**Qual é a diferença entre MAE, RMSE e MAPE?**

MAE é a média do tamanho dos erros, na unidade de trocas de óleo. RMSE dá mais peso aos erros grandes. MAPE é percentual e não é definido quando a demanda real é zero; por isso calculamos só para demanda positiva e informamos a contagem.

**A demanda zero foi retirada de toda a avaliação?**

Não. MAE e RMSE incluem todos os dias com alvo observado. Só o MAPE usa demanda positiva. Neste teste final, os 145 dias têm demanda positiva.

**Por que Random Search foi mais rápido?**

Nesta execução, avaliou seis combinações, enquanto o Grid avaliou nove, nas mesmas condições. A diferença medida foi 0,84 segundo. Uma execução não garante que essa vantagem se repita em toda máquina ou base.

**Quais parâmetros foram testados?**

No Random Forest: n_estimators = [50, 100, 200] e max_depth = [3, 5, 10]. O Grid avaliou as nove combinações; o Random amostrou seis com random_state=42. Ambos escolheram 50 árvores e profundidade 3.

**Os 13,82% e os 10,70% são o mesmo ganho?**

Não. 13,82% é a redução de MAE contra a média móvel de sete dias. 10,70% é a redução de RMSE contra o Random Forest padrão. O valor de 10,72% usa como referência o RF anterior de 200 árvores.

**Por que ficaram 724 pares?**

A planilha tem 731 dias. O último não possui a observação de amanhã para formar o alvo. Outros seis pares têm alvo ausente. Assim, 731 − 1 − 6 = 724.

**Os lags e as janelas da OAT 2 entram no modelo da OAT 3?**

A OAT 2 cria e verifica esses atributos. No experimento final da OAT 3, mantivemos as duas entradas originais: Trocas_Oleo e Manutencao_Motor de hoje. A média móvel é um baseline separado. Não devemos afirmar que lag_7 ou rolling_mean_7 alimentaram esse modelo.

**O modelo prevê os 145 dias de uma vez?**

A avaliação representa previsões de um dia à frente. Usa observações reais disponíveis até o dia anterior de cada previsão, inclusive os dias anteriores do teste. Isso é diferente de prever 145 dias futuros sem novas observações.

**Quanto a oficina economizou?**

Não medimos economia em reais. Medimos a redução do erro na quantidade prevista de trocas de óleo. Para estimar economia, precisaríamos de custos de falta, excesso de estoque e outros dados operacionais.

**De onde veio o dataset?**

Usamos a planilha fornecida na atividade. A procedência detalhada e o processo de coleta não estão documentados no repositório; não devemos inventá-los.

## Durante a apresentação

Abra o PDF ou o modo de apresentação em tela cheia. Compartilhe a janela da apresentação. As anotações ficam visíveis no editor e podem ser consultadas por quem tiver acesso ao arquivo editável; não fazem parte do PDF normal.

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
