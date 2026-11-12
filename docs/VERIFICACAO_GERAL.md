# Verificação geral do projeto MecaniQA

Revisão de **28/10/2026**. O código executou sem falhas nas verificações
abaixo. Há pendências nos documentos históricos e nas evidências das sessões.

## Execução e resultados

- **38 testes automatizados aprovados**, sem falhas, em 21,34 segundos.
- **Sete notebooks executados em kernels novos**, usando somente a planilha:
  baseline_metricas, mecaniqa_oat1, metricas_timeseries, oat3_completa,
  oat3_fechamento, oat3_gridsearch e OAT_3_Tunando_Modelo_MecaniQA.
- **Quatro scripts executados como processos separados**: app.py,
  baseline_metricas.py, pipeline.py e oat3_completa.py. Todos retornaram código 0.
- **Oito gráficos dos notebooks inspecionados**.
- O atualizador dos notebooks foi executado em uma cópia temporária e
  confirmou código equivalente nos sete notebooks. Os arquivos originais
  mantiveram suas saídas.
- Dependências sem conflitos detectados por `py -3.14 -m pip check`.
- Previsões, métricas e cenários da OAT 3 reproduzidos pelo script e pelos
  três notebooks completos, com tolerância numérica de 1e-10.
- Resultados por fold da OAT 2 reproduzidos a partir da planilha.

As execuções usaram pastas temporárias. Os resultados finais e os tempos
registrados nas apresentações permaneceram intactos.

### OAT 2

Média das métricas dos cinco folds, nas mesmas datas válidas:

| Baseline | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Naive | 6,6463 | 8,9185 | 32,6425 |
| Média móvel de 7 dias | 7,3558 | 8,0881 | 37,2182 |

Naive tem menor MAE e MAPE. A média móvel tem menor RMSE.
São 605 dias no MAE/RMSE e 604 de demanda positiva no MAPE.
O DataFrame de atributos com alvo observado tem **718 linhas válidas**.
Essa contagem não corresponde à quantidade de dias avaliados nos folds.

### OAT 3

São 724 pares válidos: 579 para treino e 145 para teste. A seleção pelo
RMSE de validação favorece Random Forest: 11,9982 contra 12,4709 do
Gradient Boosting. Grid e Random encontram 50 árvores e profundidade 3,
com o mesmo RMSE médio de validação de 10,1835.

O RF ajustado reproduz MAE 6,3321, RMSE 7,7908 e MAPE 26,8577% no teste.
A redução de MAE contra a média móvel é 13,82%; a de RMSE contra o RF
padrão é 10,70%. O RF ajustado tem o menor RMSE. O Gradient Boosting tem
menor MAE e MAPE. O MAPE do Naive também é menor que o do RF ajustado.
Não existe um vencedor em todas as métricas e não foi medido lucro em reais.

## Correspondência com as atividades

Inventário: **17 PDFs, com 275 páginas**. São cinco conjuntos de slides de
aula, oito materiais de atividade e quatro apresentações da equipe. O texto
foi extraído para consulta e os requisitos foram confrontados com o código
e as saídas. As quatro apresentações também foram renderizadas e inspecionadas.

| Etapa | Evidência no projeto | Situação |
| --- | --- | --- |
| OAT 1: leitura e indexação | Datas validadas, ordenação e DatetimeIndex | Implementado |
| OAT 1: ausências e outliers | Interpolação na EDA, IQR e limites aprendidos no treino | Implementado |
| OAT 1: decomposição | Modelo aditivo, período semanal e quatro componentes | Implementado |
| OAT 1: baselines | Naive, médias de 7/30 dias e gráfico sobreposto | Implementado |
| OAT 1: Pipeline | Imputação, IQR, escala e Random Forest com fit no treino | Implementado |
| OAT 2: baselines e métricas | ffill, shift(1), rolling(7), cinco folds e conclusão no README | Implementado |
| OAT 2: atributos | lag_1, lag_7, rolling_mean_7, rolling_std_7, head(15) e recortes temporais | Implementado |
| OAT 3: modelos padrão | Random Forest e Gradient Boosting no notebook completo | Implementado |
| OAT 3: buscas | Grid, Random com n_iter=6 e TimeSeriesSplit(gap=1) | Implementado |
| OAT 3: comparação | Três cenários, três métricas, gráfico e conclusão medida | Implementado |
| Sessões tutoriais | Cadernos preenchidos 1/2 não localizados; texto do 3 precisa de revisão | Evidência documental incompleta nesta cópia |
| Publicação | Branch local main com alterações e arquivos novos sem commit | Publicação dessas alterações não comprovada |

### Decisões metodológicas para explicar na defesa

- A interpolação e o IQR globais da EDA descrevem o histórico. Não alimentam
  a avaliação posterior como se estivessem disponíveis no passado.
- A OAT 2 cria os atributos históricos, mas avalia regras de baseline sem
  treinar um regressor nessas colunas. A OAT 3 mantém como entradas as duas
  contagens de hoje. Não afirmar que lag_7 ou rolling_std_7 alimentaram o
  modelo final da OAT 3.
- O IQR da OAT 1 não é reaplicado na OAT 2/3. As avaliações posteriores
  preservam as observações não negativas, preenchem entradas de forma causal
  e excluem alvos ausentes. Explicar essa diferença quando o professor
  mencionar a continuidade da base já limpa.
- A previsão é de um dia à frente. Observações reais anteriores dentro da
  validação podem ser usadas quando já estão disponíveis.
- rolling_std_7 usa o desvio padrão **amostral**, padrão do Pandas, com
  ddof=1. O exemplo de 08/01/2024 resulta em 8,5774. A fórmula populacional
  da aula, com divisor 7, resulta em 7,9411. As duas convenções produzem
  valores diferentes e precisam ser identificadas corretamente.

## Pendências documentais

### Cadernos preenchidos da OAT 1 e da OAT 2

As cópias de Modelos de Aprendizagem de Máquina em docs/materiais estão em
branco. A busca também verificou arquivos com nomes de cadernos em
d:/Dowloands e na pasta Downloads do usuário. As cópias encontradas dessa
disciplina continuam em branco.

Há cópias preenchidas de outra disciplina, sobre Docker, Kubernetes e
infraestrutura. Elas não foram incorporadas como evidência desta atividade.
O usuário informa que a equipe já preencheu os cadernos de ML; ainda é preciso
recuperá-los para conferir as decisões e os papéis históricos das sessões.

### Caderno preenchido da OAT 3

O original permaneceu intacto. Os seguintes trechos precisam de atenção:

| Página | Problema | Interpretação correta |
| --- | --- | --- |
| 3 | O grid limitado é apresentado como prevenção de overfitting | Limitar profundidade reduz o risco; a validação mede o desempenho. Não há garantia automática. |
| 4 | Afirma que GridSearchCV usa K-Folds embaralhados por padrão | KFold usa shuffle=False por padrão na regressão. Mesmo assim, pode treinar com períodos posteriores ao bloco validado. TimeSeriesSplit preserva a direção temporal. |
| 6 | Declara Random Search ideal e vantagem de ROI sem medir finanças | Nesta execução salva, Random avaliou seis combinações contra nove do Grid, com o mesmo erro. Tempo varia; custo em nuvem e ROI não foram medidos. |
| 8 | Deduz redução de estoque parado e falta de peças a partir do MAE | Medimos a redução do erro em quantidade de serviços. O efeito financeiro ou no estoque exige outros dados. |

Há variações nos nomes dos integrantes entre sessões, como Artur/Arthur e
Douglas Renan/Douglas Leite. A equipe precisa confirmar esses registros
antes de padronizá-los. Não foram inventados papéis ou decisões.

### Apresentações

| Arquivo | Resultado da revisão |
| --- | --- |
| mecaniQA_oat1_macapa.pdf | Histórico: objetivo incompleto, excesso de texto, títulos próximos ao logo e ausência de figuras. Descreve outliers e gráficos de baseline como pendentes, embora o notebook atual execute essas etapas. Nome de Kayky incompleto na capa. |
| mecaniQA_oat2_Macapá.pdf | Histórico: não mostra a tabela completa das três métricas nem o gráfico de baseline. Afirma 708 linhas, enquanto o código atual produz 718 no DataFrame de atributos com alvo. Traz MAE 7,04 antigo e confunde shift com prevenção de overfitting. Há títulos sobrepostos ao logo. |
| mecaniQA_oat2_Macapá-atualizada.pdf | Oito slides coerentes com os resultados reproduzidos, incluindo baselines, atributos, recortes e métricas. É a versão indicada para apresentar a OAT 2. |
| mecaniQA_oat3_macapa.pdf | Oito slides coerentes com as métricas reproduzidas. Os tempos são da execução salva em 25/10. Percentuais e escolha pela validação estão identificados. |

Todos os PDFs foram preservados nesta auditoria, conforme a preferência
do usuário. As diferenças foram registradas aqui.

### Tabelas auxiliares e reprodução

A rotina principal da OAT 3 regenera previsões, métricas, cenários,
comparação, conclusão e gráfico. A tabela auxiliar validacao_modelos.csv já
está salva, mas ainda não é regenerada automaticamente por essa função de
exportação. Seus números podem ser conferidos na saída da validação.

A OAT 2 imprime métricas e atributos, mas não exporta automaticamente todos
os arquivos de output/oat2. Seus CSVs foram conferidos nesta revisão.
Automatizar essas exportações é uma melhoria de reprodução, sem necessidade
de mudar o experimento ou as métricas apresentadas.

## Evidências e limites

- [Resumo verificável em JSON](../output/verificacao_geral.json).
- [Verificação específica da OAT 3](VERIFICACAO_OAT3.md).
- [Roteiro da OAT 2](GUIA_OAT2.md) e [roteiro da OAT 3](GUIA_OAT3.md).
- Logs, notebooks executados, extrações de PDF e imagens de inspeção em
  output/testes/auditoria_geral e output/testes/notebooks.json, ignorados no Git.
- Fontes: os PDFs de docs/aula slides, docs/materiais e docs; a planilha;
  quatro scripts principais; dois scripts de manutenção; sete notebooks;
  testes, dependências, README e guias.

Estar na branch main não comprova publicação das alterações, cumprimento
de prazos anteriores, convite ao professor ou submissão ao Blackboard. Não
foi feito commit, push, convite ou envio de formulário nesta auditoria.

Os testes demonstram execução e consistência na base atual. Não garantem
desempenho em outra oficina, ausência de qualquer defeito possível ou nota
do professor. A avaliação das sessões também depende dos documentos e da
participação da equipe, que não se deduzem da execução do código.
