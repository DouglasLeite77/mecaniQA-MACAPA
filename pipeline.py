import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Carregar e organizar os dados cronologicamente
df = pd.read_excel("./datasets/mecaniqa_dataset.xlsx")
df["Data"] = pd.to_datetime(df["Data"])
df = df.sort_values("Data").set_index("Data")

# 2. Criar o alvo: prever Trocas_Oleo do dia seguinte
dados = df[["Trocas_Oleo", "Manutencao_Motor"]].copy()
dados["Demanda_Amanha"] = dados["Trocas_Oleo"].shift(-1)
dados = dados.dropna(subset=["Demanda_Amanha"])

X = dados[["Trocas_Oleo", "Manutencao_Motor"]]
y = dados["Demanda_Amanha"]

# 3. Separar treino e teste pela ordem do tempo
# Não embaralhamos os dados de uma série temporal.
ponto_corte = int(len(dados) * 0.8)

X_train = X.iloc[:ponto_corte]
X_test = X.iloc[ponto_corte:]

y_train = y.iloc[:ponto_corte]
y_test = y.iloc[ponto_corte:]

# 4. Criar Pipeline: nulos -> escala -> modelo
pipeline = Pipeline(steps=[
    ("imputacao", SimpleImputer(strategy="median")),
    ("padronizacao", StandardScaler()),
    ("modelo", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# 5. Treinar em uma única linha
pipeline.fit(X_train, y_train)

# 6. Prever e avaliar somente nos dados futuros de teste
previsoes = pipeline.predict(X_test)
erro_mae = mean_absolute_error(y_test, previsoes)

print("Pipeline treinado com sucesso.")
print(f"Erro médio absoluto (MAE): {erro_mae:.2f}")