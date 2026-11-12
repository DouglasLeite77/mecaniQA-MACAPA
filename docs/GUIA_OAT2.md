# MecaniQA: roteiro da apresentação da OAT 2

## Como usar

Este roteiro acompanha o [PDF atualizado](mecaniQA_oat2_Macapá-atualizada.pdf).
O tempo planejado é de aproximadamente **seis minutos**, incluindo pequenas
pausas para mostrar as tabelas e o gráfico. Ensaie com cronômetro para ficar
dentro dos sete minutos exigidos. As perguntas ao final ficam para a defesa.

Use as falas como apoio e adapte as palavras ao seu jeito de falar. Os números
correspondem ao código e à planilha atuais. O PDF sem “-atualizada” permanece
na versão anterior.

## Fala por slide

### 1. Capa: cerca de 20 segundos

Professor, somos a equipe Macapá. Na OAT 2, trabalhamos com a previsão das
trocas de óleo do dia seguinte. Nosso objetivo foi comparar duas regras
simples de previsão, medir seus erros respeitando a ordem das datas e
preparar atributos que representam o histórico da oficina.

### 2. Agenda: cerca de 15 segundos

Primeiro, vamos apresentar a base e o tratamento dos dados. Depois, explicamos
os baselines, os atributos temporais e a validação. Por último, mostramos as
métricas e o que elas permitem concluir.

### 3. Dados e tratamento: cerca de 50 segundos

A planilha tem 731 dias, de janeiro de 2024 a dezembro de 2025. Ela registra
as trocas de óleo e as manutenções de motor. Encontramos nove valores
ausentes: seis em trocas de óleo e três em motor. Também existem cinco
registros negativos em motor. Como uma contagem negativa não representa uma
quantidade válida de serviços, o código marca esses valores como ausentes.

Para calcular os baselines de trocas de óleo, usamos o último valor conhecido
quando falta uma observação do histórico. O preenchimento segue do passado
para a frente. Se faltar o valor real do dia que queremos avaliar, esse dia
fica fora das métricas.

### 4. Baselines no tempo: cerca de 60 segundos

Comparamos o Naive e a média móvel de sete dias. O Naive usa a última
observação disponível antes do dia previsto. A média móvel usa os sete dias
anteriores. No código, o shift de uma linha garante esse deslocamento antes
do cálculo da janela.

Neste gráfico, a linha preta é a quantidade real, a cinza tracejada é o
Naive e a verde é a média móvel. Mostramos os últimos 60 dias para facilitar
a leitura. O Naive acompanha as mudanças com atraso. A média móvel suaviza
as oscilações, mas também demora a responder a algumas mudanças. O gráfico
ajuda a entender esse comportamento. A comparação numérica vem na tabela
de resultados.

**Ao mostrar:** aponte a legenda e uma subida ou queda. Não leia cada data.

### 5. Lags e janelas rolantes: cerca de 60 segundos

Também criamos colunas para representar o passado de forma numérica. O lag
de um dia traz a observação anterior. O lag de sete dias traz a observação
de uma semana antes. A média dos sete dias anteriores resume o nível recente,
e o desvio padrão mostra quanto esses valores variaram.

Neste exemplo, para prever 8 de janeiro de 2024, usamos somente informações
disponíveis até 7 de janeiro. O lag de um dia vale 31 e o lag de sete dias
vale 11. A média é 18,29 e o desvio é 8,58. As sete primeiras linhas ainda
não têm histórico completo, então são descartadas na preparação dos
atributos. Essas colunas ficam disponíveis para os próximos modelos.

**Ao mostrar:** destaque que 31 e 11 são observações passadas. Não são
previsões produzidas por um modelo treinado.

### 6. Validação temporal: cerca de 60 segundos

Usamos o TimeSeriesSplit com cinco divisões temporais. Em cada divisão, o
histórico de treino cresce e a validação fica em um período posterior. Não
embaralhamos as datas. Assim, cada avaliação respeita a sequência em que as
informações aparecem.

A primeira validação começa em maio de 2024. A última vai de setembro ao
fim de dezembro de 2025. Os dois baselines são avaliados nas mesmas datas
válidas de cada divisão.

Nossa avaliação representa previsões de um dia à frente. Conforme o tempo
passa, podemos usar as observações reais dos dias anteriores, inclusive
dentro da validação. Portanto, não estamos prevendo todo o período de uma
vez sem receber novas observações. Os baselines são regras fixas, sem uma
etapa de ajuste de parâmetros por fit.

### 7. Avaliação dos baselines: cerca de 75 segundos

Esta tabela apresenta a média das métricas dos cinco folds. O Naive teve
MAE de 6,65, enquanto a média móvel teve 7,36. O MAE mede o tamanho médio
do erro em trocas de óleo por dia. Por esse critério, o Naive foi melhor.

No RMSE, que dá mais peso aos erros grandes, a média móvel teve 8,09 contra
8,92 do Naive. Então, a média móvel foi melhor por esse critério.

No MAPE, o Naive teve 32,64% e a média móvel, 37,22%. Esse cálculo usa
somente dias com demanda positiva, porque não podemos dividir o erro por
uma demanda igual a zero. Foram 604 dias no MAPE e 605 no MAE e no RMSE.

A conclusão depende da métrica: o Naive teve menor MAE e MAPE, e a média
móvel teve menor RMSE. Esses valores medem erro na quantidade de serviços.
Não calculamos economia em reais nem litros de óleo.

**Ao mostrar:** aponte primeiro a coluna MAE, depois RMSE e MAPE. Faça uma
pausa antes da conclusão para deixar clara a diferença entre os critérios.

### 8. Fontes e encerramento: cerca de 20 segundos

Usamos a planilha fornecida na atividade e seguimos o enunciado e o caderno
de guias da OAT 2. O notebook, as funções e os resultados estão no
repositório. A entrega estabelece baselines avaliados no tempo e atributos
históricos para as próximas etapas. Obrigado, professor.

## Perguntas do professor

**Qual baseline foi melhor?**

Naive teve menor MAE e MAPE. A média móvel teve menor RMSE. A comparação usa
as mesmas datas em cada fold. Se a referência for o erro absoluto médio,
Naive foi melhor nesta avaliação.

**Por que a média móvel tem menor RMSE, mas maior MAE?**

As métricas pesam os erros de formas diferentes. O RMSE eleva os erros ao
quadrado e dá mais peso aos erros grandes. Por isso, a classificação pode
mudar. Aqui, comparamos a média das métricas calculadas nos cinco folds.

**Qual a diferença entre Naive e média móvel?**

Naive repete a última observação disponível antes do dia previsto. A média
móvel calcula a média dos sete dias anteriores. No código, ambos usam
preenchimento apenas com valores passados quando há lacunas no histórico.

**Como vocês evitaram vazamento de dados?**

Usamos shift(1) antes das janelas, lag_7 com shift(7), preenchimento somente
para a frente e recortes temporais sem embaralhar. O valor do dia previsto
não entra na própria previsão. Alvos ausentes ficam fora da avaliação.

**Usar valores reais anteriores dentro da validação é vazamento?**

Neste protocolo, a previsão é de um dia à frente. Antes de prever um novo
dia, a observação do dia anterior já está disponível. Isso corresponde à
operação diária que estamos simulando. Uma previsão de vários dias sem
novas observações exigiria outro protocolo.

**Por que o MAPE usa um dia a menos?**

Um dos 605 dias avaliados tem demanda zero. Esse dia continua no MAE e no
RMSE, mas fica fora do MAPE porque o cálculo divide pelo valor real.

**O MAPE de 32,64% significa 67,36% de acerto?**

Não. MAPE é uma medida de erro percentual absoluto. Subtraí-lo de 100% não
produz uma taxa de acerto de classificação nem a probabilidade de acertar a
previsão.

**O que aconteceu com as primeiras linhas dos lags?**

Precisamos de sete dias de histórico para completar todos os atributos. Na
preparação desses dados, descartamos as sete linhas iniciais sem histórico
completo e os registros sem alvo observado. A avaliação dos baselines tem
seus próprios recortes; não devemos confundir as duas contagens.

**Por que o desvio do exemplo é 8,58 e não 7,94?**

O código usa o desvio padrão amostral, padrão do Pandas em rolling.std(),
com ddof=1. O divisor é 6 para uma janela com sete valores. A fórmula
populacional apresentada na aula divide por 7 e produz aproximadamente
7,94. Os slides mostram o resultado do código: aproximadamente 8,58.

**Os lags melhoraram o resultado dos baselines?**

Não medimos esse efeito. Os baselines usam suas próprias regras. Criamos e
verificamos os atributos temporais para disponibilizá-los aos próximos
modelos. A tabela apresentada compara apenas Naive e média móvel.

**Por que não apresentar Random Forest nesta etapa?**

O foco desta OAT 2 é avaliar os baselines e preparar atributos históricos.
O experimento de regressão e a sintonia de parâmetros aparecem na OAT 3,
com outro protocolo de avaliação.

**Quantos litros de óleo ou reais o erro representa?**

Nosso alvo é a quantidade de trocas de óleo por dia. Para converter em litros
ou reais, precisaríamos de volume por serviço e custos operacionais, que
não fazem parte deste cálculo.

**De onde veio a base?**

Usamos a planilha fornecida na atividade. A procedência detalhada e o
processo de coleta não estão documentados no repositório.

## Mensagem para o grupo

Pessoal, alguém tem os cadernos de guias da OAT 1 e da OAT 2 preenchidos pela
nossa equipe? Os que tenho aqui estão em branco. Se tiverem, mandem os
arquivos, por favor.
