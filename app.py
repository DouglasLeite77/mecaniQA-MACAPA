"""Decomposição exploratória da série de trocas de óleo (OAT 1)."""

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from pipeline import ler_planilha


def criar_decomposicao():
    df = ler_planilha()
    # A interpolação usa a série completa apenas para análise exploratória.
    serie = df["Trocas_Oleo"].interpolate(method="time").ffill().bfill()
    resultado = seasonal_decompose(serie, model="additive", period=7)
    figura = resultado.plot()
    figura.set_size_inches(10, 8)
    for eixo, titulo in zip(figura.axes, ["Série observada", "Tendência", "Sazonalidade", "Ruído"]):
        eixo.set_title(titulo)
    figura.tight_layout()
    return figura


def main():
    figura = criar_decomposicao()
    plt.show()
    return figura


if __name__ == "__main__":
    main()
