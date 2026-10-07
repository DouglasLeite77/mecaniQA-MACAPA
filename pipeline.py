from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARQUIVO_DADOS = Path(__file__).resolve().parent / "datasets" / "mecaniqa_dataset.xlsx"


def carregar_dados():
    df = pd.read_excel(ARQUIVO_DADOS)
    df["Data"] = pd.to_datetime(df["Data"], errors="raise")
    df = df.sort_values("Data").set_index("Data")

    if df.index.hasnans or df.index.has_duplicates:
        raise ValueError("A planilha precisa ter datas válidas e sem duplicatas.")
    if not df.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("A planilha precisa ter uma linha por dia, sem intervalos.")

    # Quantidades negativas são entradas inválidas; a imputação ocorre no pipeline.
    colunas = ["Trocas_Oleo", "Manutencao_Motor"]
    df[colunas] = df[colunas].mask(df[colunas] < 0)
    dados = df[colunas].copy()
    dados["Demanda_Amanha"] = df["Trocas_Oleo"].shift(-1)

    # Previsões de amanhã usando somente informações disponíveis até hoje.
    dados["Naive"] = df["Trocas_Oleo"].ffill()
    dados["Media_Movel_7"] = df["Trocas_Oleo"].ffill().rolling(7, min_periods=1).mean()
    dados = dados.dropna(subset=["Demanda_Amanha"])
    return dados, colunas


def criar_pipeline(modelo):
    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler()),
        ("modelo", modelo),
    ])


def calcular_metricas(y_real, previsao):
    real = np.asarray(y_real, dtype=float)
    previsto = np.asarray(previsao, dtype=float)
    positivos = real > 0
    mape = (
        np.mean(np.abs((real[positivos] - previsto[positivos]) / real[positivos])) * 100
        if positivos.any() else np.nan
    )
    return {
        "MAE": mean_absolute_error(real, previsto),
        "RMSE": np.sqrt(mean_squared_error(real, previsto)),
        "MAPE (demanda > 0) %": mape,
    }


def main():
    dados, colunas = carregar_dados()
    corte = int(len(dados) * 0.8)
    treino, teste = dados.iloc[:corte], dados.iloc[corte:]
    X_train, y_train = treino[colunas], treino["Demanda_Amanha"]
    X_test, y_test = teste[colunas], teste["Demanda_Amanha"]

    print(f"Exemplos com alvo válido: {len(dados)}")
    print(f"Treino: {len(treino)} | Teste: {len(teste)}")
    print(f"Datas de referência do treino: {treino.index.min().date()} a {treino.index.max().date()}")
    print(f"Datas de referência do teste: {teste.index.min().date()} a {teste.index.max().date()}")

    # O intervalo protege a fronteira dos folds, pois o alvo é do dia seguinte.
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    modelos = {
        "Random Forest": criar_pipeline(RandomForestRegressor(n_estimators=200, random_state=42)),
        "Gradient Boosting": criar_pipeline(GradientBoostingRegressor(random_state=42)),
    }
    grids = {
        "Random Forest": {
            "modelo__n_estimators": [100, 200, 300],
            "modelo__max_depth": [5, 10, None],
        },
        "Gradient Boosting": {
            "modelo__n_estimators": [100, 200, 300],
            "modelo__max_depth": [2, 3],
            "modelo__learning_rate": [0.05, 0.1],
        },
    }

    inicio = perf_counter()
    resultados_cv = {}
    for nome, modelo in modelos.items():
        scores = cross_val_score(
            modelo, X_train, y_train, cv=tscv,
            scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise",
        )
        resultados_cv[nome] = -scores.mean()
        print(f"{nome} inicial - RMSE médio na validação: {resultados_cv[nome]:.4f}", flush=True)

    # A escolha usa apenas a validação dentro dos 80% de treino.
    vencedor = min(resultados_cv, key=resultados_cv.get)
    grid = grids[vencedor]
    print(f"\nModelo escolhido: {vencedor}")
    print(f"Combinações: {len(ParameterGrid(grid))} | Folds: {tscv.n_splits}", flush=True)
    busca = GridSearchCV(
        modelos[vencedor], param_grid=grid, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=1,
        refit=True, error_score="raise",
    )
    busca.fit(X_train, y_train)
    print("best_params_:", busca.best_params_)
    print(f"RMSE médio na validação após ajuste: {-busca.best_score_:.4f}")

    # O teste final é usado somente depois da escolha e do ajuste.
    previsoes = pd.DataFrame(index=y_test.index + pd.Timedelta(days=1))
    previsoes.index.name = "Data_da_previsao"
    previsoes["Real"] = y_test.to_numpy()
    previsoes["Naive"] = teste["Naive"].to_numpy()
    previsoes["Média móvel 7 dias"] = teste["Media_Movel_7"].to_numpy()
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        previsoes[f"{nome} inicial"] = modelo.predict(X_test)
    nome_ajustado = f"{vencedor} ajustado"
    previsoes[nome_ajustado] = busca.predict(X_test)

    metricas = pd.DataFrame({
        nome: calcular_metricas(previsoes["Real"], previsoes[nome])
        for nome in previsoes.columns if nome != "Real"
    }).T
    metricas.index.name = "Modelo"
    print("\nMétricas no mesmo teste final:")
    print(metricas.round(4).to_string())
    print(f"MAPE calculado em {int((y_test > 0).sum())} de {len(y_test)} dias; demanda zero não tem MAPE definido.")

    rmse_inicial = metricas.loc[f"{vencedor} inicial", "RMSE"]
    rmse_ajustado = metricas.loc[nome_ajustado, "RMSE"]
    reducao = (rmse_inicial - rmse_ajustado) / rmse_inicial * 100 if rmse_inicial else np.nan
    print(f"\nRedução percentual do RMSE no teste: {reducao:.2f}%")
    if not np.isfinite(reducao):
        print("O erro inicial é zero; a redução percentual não é definida.")
    elif reducao < 0:
        print("O ajuste aumentou o erro no teste. A melhora na validação não se confirmou.")
    elif reducao == 0:
        print("O ajuste manteve o erro do modelo inicial.")
    print(f"Tempo total: {perf_counter() - inicio:.2f} segundos")

    return {
        "metricas": metricas,
        "previsoes": previsoes,
        "busca": busca,
        "rmse_cv_inicial": resultados_cv,
        "modelo_escolhido": vencedor,
        "reducao_rmse_percentual": reducao,
    }


if __name__ == "__main__":
    main()
