import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import app
import baseline_metricas as b
import pipeline as p


def test_baselines_usam_apenas_dias_anteriores():
    serie = pd.Series([10., 20., np.nan, 40., 50., 60., 70., 80.],
                      index=pd.date_range("2025-01-01", periods=8))
    assert b.modelo_naive(serie).iloc[3] == 20
    assert b.modelo_media_movel(serie).iloc[7] == pytest.approx(270 / 7)
    assert b.modelo_media_movel(serie).iloc[:7].isna().all()
    features = b.criar_features_temporais(serie)
    assert features.loc["2025-01-08", "lag_7"] == 10
    assert features.loc["2025-01-08", "lag_1"] == 70
    alterada = serie.copy()
    alterada.iloc[7] = 9999
    pd.testing.assert_frame_equal(features, b.criar_features_temporais(alterada))


def test_baselines_com_mesmos_dias_e_mape_definido():
    serie = p.ler_planilha()["Trocas_Oleo"]
    tabela = b.avaliar_baselines(serie)
    assert len(tabela) == 10
    assert tabela.groupby("Fold").Dias.nunique().eq(1).all()
    assert tabela.groupby("Fold")["Dias MAPE"].nunique().eq(1).all()
    assert np.isfinite(tabela[["MAE", "RMSE", "MAPE (demanda > 0) %"]]).all().all()
    assert tabela["MAPE (demanda > 0) %"].lt(1000).all()


@pytest.mark.parametrize("real,pred", [
    ([], []), ([1, 2], [1]), ([[1]], [[1]]),
    ([np.nan], [1]), ([1], [np.inf]), ([-1], [1]),
])
def test_metricas_rejeitam_entradas_invalidas(real, pred):
    with pytest.raises(ValueError):
        p.calcular_metricas(real, pred)


def test_metricas_rejeitam_datas_desalinhadas():
    with pytest.raises(ValueError, match="mesmas datas"):
        p.calcular_metricas(pd.Series([1], index=["2025-01-01"]),
                            pd.Series([1], index=["2025-01-02"]))


def test_leitura_rejeita_colunas_ausentes(monkeypatch):
    monkeypatch.setattr(p.pd, "read_excel", lambda *_: pd.DataFrame({"Data": [pd.Timestamp("2025-01-01")]}))
    with pytest.raises(ValueError, match="Colunas ausentes"):
        p.ler_planilha()


def test_leitura_rejeita_planilha_vazia(monkeypatch):
    monkeypatch.setattr(p.pd, "read_excel", lambda *_: pd.DataFrame(columns=["Data", "Trocas_Oleo", "Manutencao_Motor"]))
    with pytest.raises(ValueError, match="vazia"):
        p.ler_planilha()


def test_importar_scripts_nao_treina_nem_abre_graficos(monkeypatch):
    def inesperado(*args, **kwargs):
        pytest.fail("Importar um módulo não deve ler dados ou abrir janelas.")
    monkeypatch.setattr(p.pd, "read_excel", inesperado)
    monkeypatch.setattr(plt, "show", inesperado)
    importlib.reload(app)
    importlib.reload(b)


def test_graficos_exploratorios():
    fig = app.criar_decomposicao()
    assert len(fig.axes) == 4
    plt.close(fig)
    serie = p.ler_planilha()["Trocas_Oleo"]
    fig = b.criar_grafico_baselines(serie)
    assert len(fig.axes[0].lines) == 3
    assert all(len(line.get_ydata()) == 60 for line in fig.axes[0].lines)
    plt.close(fig)
