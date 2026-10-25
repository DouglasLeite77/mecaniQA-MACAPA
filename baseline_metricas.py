"""Baselines, validação temporal e atributos históricos da OAT 2."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from pipeline import calcular_metricas, ler_planilha


def modelo_naive(serie):
    """Prevê o dia atual com a última observação disponível até ontem."""
    return serie.ffill().shift(1)


def modelo_media_movel(serie, janela=7):
    """Usa uma janela completa de dias anteriores à data prevista."""
    return serie.ffill().shift(1).rolling(window=janela).mean()


def criar_features_temporais(serie):
    passado = serie.ffill().shift(1)
    return pd.DataFrame({
        "lag_1": passado,
        "lag_7": serie.ffill().shift(7),
        "rolling_mean_7": passado.rolling(7).mean(),
        "rolling_std_7": passado.rolling(7).std(),
    })


def avaliar_baselines(serie, n_splits=5):
    """Compara os baselines nas mesmas datas válidas de cada fold."""
    previsoes = pd.DataFrame({
        "Real": serie,
        "Naive": modelo_naive(serie),
        "Média móvel 7 dias": modelo_media_movel(serie),
    })
    resultados = []
    for fold, (_, indices_teste) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(serie), 1):
        teste = previsoes.iloc[indices_teste].dropna()
        if teste.empty:
            raise ValueError(f"O fold {fold} não tem dias válidos para avaliar os dois baselines.")
        for nome in ["Naive", "Média móvel 7 dias"]:
            resultados.append({
                "Fold": fold,
                "Modelo": nome,
                "Dias": len(teste),
                "Dias MAPE": int(teste["Real"].gt(0).sum()),
                **calcular_metricas(teste["Real"], teste[nome]),
            })
    return pd.DataFrame(resultados)


def criar_grafico_baselines(serie):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(serie.tail(60), label="Trocas de óleo reais", color="black", linewidth=2)
    ax.plot(modelo_naive(serie).tail(60), label="Naive (dia anterior)", linestyle="--")
    ax.plot(modelo_media_movel(serie).tail(60), label="Média móvel 7 dias", linestyle="-.")
    ax.set(title="MecaniQA - baselines nos últimos 60 dias", xlabel="Data prevista", ylabel="Trocas de óleo")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main():
    df = ler_planilha()
    serie = df["Trocas_Oleo"]
    print(f"Dados carregados: {len(df)} dias.")
    resultados = avaliar_baselines(serie)
    resumo = resultados.groupby("Modelo")[["MAE", "RMSE", "MAPE (demanda > 0) %"]].mean()
    print("Média das métricas nos cinco folds temporais:")
    print(resumo.round(4).to_string())
    print("MAPE: somente dias de demanda positiva. Quantidades por fold:")
    print(resultados[["Fold", "Modelo", "Dias", "Dias MAPE"]].to_string(index=False))
    features = criar_features_temporais(serie).dropna()
    print("Atributos temporais (primeiras 15 linhas válidas):")
    print(features.head(15).to_string())
    figura = criar_grafico_baselines(serie)
    plt.show()
    return resultados, features, figura


if __name__ == "__main__":
    main()
