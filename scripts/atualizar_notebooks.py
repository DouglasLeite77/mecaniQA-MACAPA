"""Atualiza notebooks autossuficientes a partir dos scripts do terminal."""

import ast
import copy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def codigo_modulo(nome, sem_main=True):
    texto = (ROOT / nome).read_text(encoding="utf-8")
    removidos = []
    for node in ast.parse(texto).body:
        if isinstance(node, ast.ImportFrom) and node.module in {"pipeline", "oat3_completa"}:
            removidos.append(node)
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            removidos.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "main" and sem_main:
            removidos.append(node)
    linhas = texto.splitlines(keepends=True)
    for node in sorted(removidos, key=lambda n: n.lineno, reverse=True):
        del linhas[node.lineno - 1:node.end_lineno]
    texto = "".join(linhas).strip()
    texto = texto.replace('Path(__file__).resolve().parent / "datasets" / "mecaniqa_dataset.xlsx"', 'localizar_planilha()')
    texto = texto.replace('Path(__file__).resolve().parent / "output" / "oat3"', 'Path("output/oat3")')
    return texto


def salvar(nome, nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, ROOT / nome)


def main():
    localizar = '''from pathlib import Path


def localizar_planilha():
    for caminho in [Path("datasets/mecaniqa_dataset.xlsx"), Path("mecaniqa_dataset.xlsx")]:
        if caminho.is_file():
            return caminho
    try:
        from google.colab import files
    except ImportError as erro:
        raise FileNotFoundError("Coloque mecaniqa_dataset.xlsx em datasets ou ao lado do notebook.") from erro
    print("Selecione mecaniqa_dataset.xlsx.")
    files.upload()
    caminho = Path("mecaniqa_dataset.xlsx")
    if not caminho.is_file():
        raise FileNotFoundError("Envie o arquivo com o nome mecaniqa_dataset.xlsx.")
    return caminho
'''
    comum = localizar + "\n\n" + codigo_modulo("pipeline.py")
    completa = nbformat.read(ROOT / "oat3_completa.ipynb", as_version=4)
    completa.cells[2].source = comum + "\n\n" + codigo_modulo("oat3_completa.py")
    salvar("oat3_completa.ipynb", completa)

    grid = nbformat.read(ROOT / "oat3_gridsearch.ipynb", as_version=4)
    grid.cells[2].source = localizar + "\n\n" + codigo_modulo("pipeline.py", sem_main=False) + "\n\nresultado = main()"
    salvar("oat3_gridsearch.ipynb", grid)

    # Mantém os nomes já compartilhados pela equipe, usando o mesmo experimento.
    for nome, titulo in [
        ("oat3_fechamento.ipynb", "MecaniQA - fechamento da OAT 3"),
        ("OAT_3_Tunando_Modelo_MecaniQA.ipynb", "MecaniQA - tunando o modelo"),
    ]:
        nb = copy.deepcopy(completa)
        nb.cells[0].source = f"# {titulo}\n\nEsta versão executa o mesmo experimento de oat3_completa.ipynb, com a planilha real.\n\n" + "\n".join(completa.cells[0].source.splitlines()[2:])
        salvar(nome, nb)

    baselines = codigo_modulo("baseline_metricas.py")
    for nome, titulo in [
        ("baseline_metricas.ipynb", "MecaniQA - baselines"),
        ("metricas_timeseries.ipynb", "MecaniQA - OAT 2: métricas e atributos temporais"),
    ]:
        cells = [
            nbformat.v4.new_markdown_cell(f"# {titulo}\n\nAvaliação de Naive e média móvel nas mesmas datas de cada fold temporal. O preenchimento usa apenas observações anteriores; os alvos ausentes não são imputados. O MAPE exclui demanda zero. As métricas desta validação não devem ser comparadas diretamente ao teste final da OAT 3."),
            nbformat.v4.new_code_cell(comum + "\n\n" + baselines),
            nbformat.v4.new_code_cell('df = ler_planilha()\nserie = df["Trocas_Oleo"]\nprint(f"Base: {len(df)} dias")\ndisplay(df.head())'),
            nbformat.v4.new_markdown_cell("## Validação temporal\n\nOs baselines são regras sem ajuste de parâmetros. Cada fold testa o período posterior ao treino. A previsão de cada dia pode usar as observações dos dias anteriores do próprio teste."),
            nbformat.v4.new_code_cell('resultados = avaliar_baselines(serie)\ndisplay(resultados.round(4))\nresumo = resultados.groupby("Modelo")[["MAE", "RMSE", "MAPE (demanda > 0) %"]].mean()\nprint("Média dos cinco folds:")\ndisplay(resumo.round(4))\nfor nome, linha in resumo.iterrows():\n    print(f"Resultados do Baseline ({nome}) - MAE: {linha.iloc[0]:.2f}, RMSE: {linha.iloc[1]:.2f}, MAPE (demanda > 0): {linha.iloc[2]:.2f}%")'),
            nbformat.v4.new_code_cell('figura = criar_grafico_baselines(serie)\nplt.show()'),
            nbformat.v4.new_markdown_cell("## Lags e janelas rolantes\n\nlag_1 e lag_7 representam observações de um e sete dias antes do alvo. Média e desvio padrão usam os sete dias anteriores. As sete primeiras linhas servem de aquecimento; os vazios iniciais são descartados. Os atributos ficam disponíveis para os recortes do TimeSeriesSplit sem usar o valor do dia previsto."),
            nbformat.v4.new_code_cell('features = criar_features_temporais(serie)\ndados_features = pd.concat([serie.rename("Alvo"), features], axis=1).dropna()\ndisplay(dados_features.head(15))\nfor fold, (treino, teste) in enumerate(TimeSeriesSplit(n_splits=5).split(dados_features), 1):\n    print(f"Fold {fold}: treino até {dados_features.index[treino[-1]].date()} | teste a partir de {dados_features.index[teste[0]].date()}")'),
        ]
        nb = nbformat.v4.new_notebook(cells=cells)
        for i, cell in enumerate(nb.cells):
            cell.id = f"oat2-cell-{i:02d}"
        salvar(nome, nb)


if __name__ == "__main__":
    main()
