from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)

from pipeline import carregar_dados, criar_pipeline, calcular_metricas


GRIDS = {
    "Random Forest": {
        "modelo__n_estimators": [50, 100, 200],
        "modelo__max_depth": [3, 5, 10],
    },
    "Gradient Boosting": {
        "modelo__n_estimators": [100, 200, 300],
        "modelo__max_depth": [2, 3],
        "modelo__learning_rate": [0.05, 0.1],
    },
}
N_ITER = 6
PASTA_RESULTADOS = Path(__file__).resolve().parent / "output" / "oat3"


def preparar_experimento():
    dados, colunas = carregar_dados()
    corte = int(len(dados) * 0.8)
    treino, teste = dados.iloc[:corte], dados.iloc[corte:]
    cv = TimeSeriesSplit(n_splits=5, gap=1)

    # Um alvo em t+1 deve estar disponível antes da primeira referência da validação.
    for indices_treino, indices_validacao in cv.split(treino):
        ultima_data_alvo = treino.index[indices_treino[-1]] + pd.Timedelta(days=1)
        primeira_referencia = treino.index[indices_validacao[0]]
        if ultima_data_alvo >= primeira_referencia:
            raise ValueError("O fold não respeita a disponibilidade temporal do alvo.")

    print(f"Exemplos válidos: {len(dados)} | Treino: {len(treino)} | Teste: {len(teste)}")
    print("Previsão diária de amanhã com informações disponíveis até hoje.")
    print("Escolha dos modelos e buscas: somente no treino, em cinco folds temporais.")
    return {
        "X_train": treino[colunas],
        "y_train": treino["Demanda_Amanha"],
        "X_test": teste[colunas],
        "y_test": teste["Demanda_Amanha"],
        "teste": teste,
        "cv": cv,
    }


def comparar_modelos(experimento):
    # Apenas a semente é fixada; os hiperparâmetros de aprendizado são os padrões.
    modelos = {
        "Random Forest": criar_pipeline(RandomForestRegressor(random_state=42)),
        "Gradient Boosting": criar_pipeline(GradientBoostingRegressor(random_state=42)),
    }
    rmse_cv = {}
    for nome, modelo in modelos.items():
        scores = cross_val_score(
            modelo,
            experimento["X_train"],
            experimento["y_train"],
            cv=experimento["cv"],
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )
        rmse_cv[nome] = -scores.mean()
        print(f"{nome} padrão - RMSE médio na validação: {rmse_cv[nome]:.4f}")
    escolhido = min(rmse_cv, key=rmse_cv.get)
    print("Modelo escolhido pela validação:", escolhido)
    return modelos, escolhido, rmse_cv


def executar_grid(experimento, modelo, grid):
    busca = GridSearchCV(
        modelo,
        param_grid=grid,
        cv=experimento["cv"],
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    inicio = perf_counter()
    busca.fit(experimento["X_train"], experimento["y_train"])
    segundos = perf_counter() - inicio
    print(f"Grid Search: {len(ParameterGrid(grid))} combinações, {segundos:.2f} segundos")
    print("best_params_:", busca.best_params_)
    print(f"RMSE médio na validação: {-busca.best_score_:.4f}")
    return {"busca": busca, "segundos": segundos, "candidatos": len(ParameterGrid(grid))}


def executar_random(experimento, modelo, grid, n_iter=N_ITER):
    total = len(ParameterGrid(grid))
    if not isinstance(n_iter, int) or not 1 <= n_iter <= total:
        raise ValueError(f"n_iter deve ser inteiro entre 1 e {total}.")
    busca = RandomizedSearchCV(
        modelo,
        param_distributions=grid,
        n_iter=n_iter,
        cv=experimento["cv"],
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    inicio = perf_counter()
    busca.fit(experimento["X_train"], experimento["y_train"])
    segundos = perf_counter() - inicio
    print(f"Random Search: n_iter={n_iter}, {segundos:.2f} segundos")
    print("best_params_:", busca.best_params_)
    print(f"RMSE médio na validação: {-busca.best_score_:.4f}")
    return {"busca": busca, "segundos": segundos, "candidatos": n_iter}


def reducao_percentual(erro_inicial, erro_final):
    if erro_inicial == 0:
        return float("nan")
    return (erro_inicial - erro_final) / erro_inicial * 100


def avaliar_teste(experimento, modelos, escolhido, grid, random):
    metodos = {"Grid Search": grid, "Random Search": random}
    # Empates de validação mantêm o Grid, sem consultar o teste final.
    melhor_metodo = min(metodos, key=lambda nome: -metodos[nome]["busca"].best_score_)
    nome_final = f"{escolhido} - {melhor_metodo}"

    y_test = experimento["y_test"]
    previsoes = pd.DataFrame(index=y_test.index + pd.Timedelta(days=1))
    previsoes.index.name = "Data_da_previsao"
    previsoes["Real"] = y_test.to_numpy()
    previsoes["Naive"] = experimento["teste"]["Naive"].to_numpy()
    previsoes["Média móvel 7 dias"] = experimento["teste"]["Media_Movel_7"].to_numpy()

    for nome, modelo in modelos.items():
        modelo.fit(experimento["X_train"], experimento["y_train"])
        previsoes[f"{nome} padrão"] = modelo.predict(experimento["X_test"])
    referencia = criar_pipeline(RandomForestRegressor(n_estimators=200, random_state=42))
    referencia.fit(experimento["X_train"], experimento["y_train"])
    previsoes["Random Forest anterior (200 árvores)"] = referencia.predict(experimento["X_test"])
    for nome, resultado in metodos.items():
        previsoes[f"{escolhido} - {nome}"] = resultado["busca"].predict(experimento["X_test"])

    metricas = pd.DataFrame({
        nome: calcular_metricas(previsoes["Real"], previsoes[nome])
        for nome in previsoes.columns if nome != "Real"
    }).T.rename_axis("Modelo")
    comparacao = pd.DataFrame({
        nome: {
            "Tempo (s)": resultado["segundos"],
            "Combinações": resultado["candidatos"],
            "RMSE validação": -resultado["busca"].best_score_,
            "RMSE teste": metricas.loc[f"{escolhido} - {nome}", "RMSE"],
        }
        for nome, resultado in metodos.items()
    }).T.rename_axis("Método")

    cenarios = metricas.loc[["Média móvel 7 dias", f"{escolhido} padrão", nome_final]].copy()
    cenarios.index = ["Baseline antigo (MM7)", "ML inicial padrão", f"ML ajustado ({melhor_metodo})"]
    rmse_grid = metricas.loc[f"{escolhido} - Grid Search", "RMSE"]
    ganho_referencia = reducao_percentual(
        metricas.loc["Random Forest anterior (200 árvores)", "RMSE"], rmse_grid
    )
    ganho_inicial = reducao_percentual(
        metricas.loc[f"{escolhido} padrão", "RMSE"], metricas.loc[nome_final, "RMSE"]
    )
    print("\nMétricas no mesmo teste final:")
    print(metricas.round(4).to_string())
    print("\nTempo e erro das duas buscas:")
    print(comparacao.round(4).to_string())
    print(f"\nMétodo escolhido pela validação: {melhor_metodo}")
    print(f"Redução do RMSE ajustado versus inicial padrão: {ganho_inicial:.2f}%")
    print(f"Redução do RMSE do Grid versus referência anterior (200 árvores): {ganho_referencia:.2f}%")
    print(f"MAPE: {int((y_test > 0).sum())} de {len(y_test)} dias com demanda maior que zero.")
    return {
        "metricas": metricas,
        "cenarios": cenarios,
        "comparacao": comparacao,
        "previsoes": previsoes,
        "modelo_escolhido": escolhido,
        "melhor_metodo": melhor_metodo,
        "nome_final": nome_final,
        "reducao_inicial_percentual": ganho_inicial,
        "reducao_referencia_percentual": ganho_referencia,
        "grid": grid,
        "random": random,
    }


def conclusao_markdown(resultado):
    c = resultado["comparacao"]
    g, r = c.loc["Grid Search"], c.loc["Random Search"]
    diferenca_tempo = g["Tempo (s)"] - r["Tempo (s)"]
    diferenca_erro = r["RMSE teste"] - g["RMSE teste"]
    if diferenca_tempo > 0:
        texto_tempo = f"O Random Search levou {diferenca_tempo:.2f} segundos a menos."
    else:
        texto_tempo = f"O Random Search levou {-diferenca_tempo:.2f} segundos a mais."
    texto_erro = (
        f"Seu RMSE no teste foi {diferenca_erro:.4f} maior."
        if diferenca_erro > 0 else
        f"Seu RMSE no teste foi {-diferenca_erro:.4f} menor."
        if diferenca_erro < 0 else "As duas buscas tiveram o mesmo RMSE no teste."
    )
    mae_mm7 = resultado["metricas"].loc["Média móvel 7 dias", "MAE"]
    mae_final = resultado["metricas"].loc[resultado["nome_final"], "MAE"]
    return (
        "# MecaniQA - conclusão da OAT 3\n\n"
        f"O modelo escolhido na validação foi **{resultado['modelo_escolhido']}**. "
        "As duas buscas utilizaram o mesmo espaço de parâmetros, cinco folds de "
        "TimeSeriesSplit com gap=1, o mesmo RMSE e uma única tarefa de processamento. "
        "Os tempos abaixo medem somente o fit de cada busca, incluindo o refit.\n\n"
        f"- Grid Search: {int(g['Combinações'])} combinações, {g['Tempo (s)']:.2f} segundos, "
        f"RMSE de validação {g['RMSE validação']:.4f} e RMSE de teste {g['RMSE teste']:.4f}.\n"
        f"- Random Search: {int(r['Combinações'])} combinações, {r['Tempo (s)']:.2f} segundos, "
        f"RMSE de validação {r['RMSE validação']:.4f} e RMSE de teste {r['RMSE teste']:.4f}.\n\n"
        f"{texto_tempo} {texto_erro} A escolha final foi **{resultado['melhor_metodo']}**, "
        "pelo RMSE na validação, antes de consultar o teste; em caso de empate, "
        "mantemos o Grid conforme a regra definida no experimento.\n\n"
        "Random Search é uma opção quando o espaço de parâmetros é grande e há limite "
        "de processamento. Nesta base pequena, a diferença de tempo é uma medição desta "
        "execução; é necessário observar conjuntamente o custo e o erro, sem garantir "
        "que uma busca será sempre mais rápida ou mais precisa.\n\n"
        f"O RMSE final mudou {resultado['reducao_inicial_percentual']:.2f}% em relação ao "
        "modelo inicial padrão (percentual positivo indica redução; negativo indica piora). "
        f"O Grid reduziu o RMSE em {resultado['reducao_referencia_percentual']:.2f}% frente "
        "ao Random Forest anterior de 200 árvores, reexecutado nas mesmas condições.\n\n"
        "Para explicar o resultado ao dono da oficina, o MAE é a medida mais direta: "
        f"a média móvel errou {mae_mm7:.2f} trocas de óleo por dia, enquanto o modelo "
        f"ajustado errou {mae_final:.2f}. O RMSE destaca erros maiores. O MAPE considera "
        "somente dias com demanda positiva e deve ser lido junto das outras métricas. "
        "Esses erros são quantidades de serviços; não medem lucro em reais sem custos "
        "de falta e excesso de estoque.\n\n"
        "O teste final contém 145 dias. Esta avaliação prevê um dia por vez usando "
        "observações disponíveis até o dia anterior à previsão; não é uma previsão "
        "de todo o período de teste de uma única vez.\n"
    )


def criar_grafico(resultado):
    dados = resultado["previsoes"].tail(60)
    fig, ax = plt.subplots(figsize=(13, 5))
    curvas = {
        "Real": "Trocas de óleo reais",
        "Média móvel 7 dias": "Baseline: média móvel 7 dias",
        f"{resultado['modelo_escolhido']} padrão": "Modelo ML inicial",
        resultado["nome_final"]: "Modelo ML ajustado",
    }
    for coluna, legenda in curvas.items():
        ax.plot(dados.index, dados[coluna], label=legenda, linewidth=2 if coluna == "Real" else 1.5)
    ax.set_title("MecaniQA - comparação no teste final (últimos 60 dias)")
    ax.set_xlabel("Data prevista")
    ax.set_ylabel("Quantidade de trocas de óleo")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def salvar_resultados(resultado, figura, pasta=PASTA_RESULTADOS):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for nome in ["metricas", "cenarios", "comparacao", "previsoes"]:
        resultado[nome].to_csv(pasta / f"{nome}.csv", encoding="utf-8-sig")
    figura.savefig(pasta / "comparacao_final.png", dpi=160)
    (pasta / "conclusao.md").write_text(conclusao_markdown(resultado), encoding="utf-8")
    print("Resultados salvos em:", pasta.resolve())


def main():
    experimento = preparar_experimento()
    modelos, escolhido, _ = comparar_modelos(experimento)
    grid = executar_grid(experimento, modelos[escolhido], GRIDS[escolhido])
    random = executar_random(experimento, modelos[escolhido], GRIDS[escolhido])
    resultado = avaliar_teste(experimento, modelos, escolhido, grid, random)
    figura = criar_grafico(resultado)
    salvar_resultados(resultado, figura)
    plt.close(figura)
    print("\n" + conclusao_markdown(resultado))
    return resultado


if __name__ == "__main__":
    main()
