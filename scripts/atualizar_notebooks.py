"""Atualiza os notebooks autossuficientes a partir dos módulos em src/.

Os notebooks repetem o código dos módulos para rodarem sozinhos (inclusive no
Colab). Depois de alterar algo em src/, execute este script e, em seguida,
``scripts/testar_notebooks.py --salvar`` para gravar as saídas novamente.

Estrutura:
    notebooks/oat3_completa.ipynb        experimento completo da OAT 3
    notebooks/metricas_timeseries.ipynb  baselines e atributos da OAT 2
    notebooks/legado/*.ipynb             nomes antigos, mantidos para links já compartilhados
    notebooks/mecaniqa_oat1.ipynb        análise da OAT 1 (mantida à parte; só o localizador é atualizado)
"""

import ast
import copy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NOTEBOOKS = ROOT / "notebooks"
LEGADO = NOTEBOOKS / "legado"

# Funções comuns a todos os notebooks: encontram a raiz do projeto de qualquer pasta
# (notebooks/, notebooks/legado/ ou cópia temporária) e, no Colab, pedem o upload da planilha.
LOCALIZADOR = '''from pathlib import Path


def pasta_projeto():
    """Raiz do projeto (pasta que contém datasets/); no Colab, a pasta atual."""
    for pasta in [Path.cwd(), *Path.cwd().parents]:
        if (pasta / "datasets").is_dir():
            return pasta
    return Path.cwd()


def localizar_planilha():
    """Caminho da planilha; no Colab, solicita o envio do arquivo."""
    for caminho in [pasta_projeto() / "datasets" / "mecaniqa_dataset.xlsx", Path("mecaniqa_dataset.xlsx")]:
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

# Versão anterior do localizador, ainda presente no notebook da OAT 1.
LOCALIZADOR_ANTIGO_OAT1 = '''def localizar_planilha():
    for caminho in [Path("datasets/mecaniqa_dataset.xlsx"), Path("mecaniqa_dataset.xlsx")]:'''


def codigo_modulo(nome, sem_main=True):
    """Texto do módulo sem imports internos, sem o bloco __main__ e (opcionalmente) sem main()."""
    texto = (SRC / nome).read_text(encoding="utf-8")
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
    # Nos módulos os caminhos partem de src/; nos notebooks, da raiz encontrada em tempo de execução.
    for antigo, novo in [
        ('Path(__file__).resolve().parents[1] / "datasets" / "mecaniqa_dataset.xlsx"', "localizar_planilha()"),
        ('Path(__file__).resolve().parents[1] / "output" / "oat3"', 'pasta_projeto() / "output" / "oat3"'),
    ]:
        texto = texto.replace(antigo, novo)
    if "__file__" in texto:
        raise ValueError(f"{nome}: caminho com __file__ não convertido; os notebooks não o reconhecem.")
    return texto


def salvar(caminho, nb):
    """Grava o notebook sem saídas antigas (rode testar_notebooks.py --salvar depois)."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    caminho.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, caminho)
    print("Atualizado:", caminho.relative_to(ROOT))


def atualizar_oat3():
    comum = LOCALIZADOR + "\n\n" + codigo_modulo("pipeline.py")
    completa = nbformat.read(NOTEBOOKS / "oat3_completa.ipynb", as_version=4)
    completa.cells[2].source = comum + "\n\n" + codigo_modulo("oat3_completa.py")
    salvar(NOTEBOOKS / "oat3_completa.ipynb", completa)

    grid = nbformat.read(LEGADO / "oat3_gridsearch.ipynb", as_version=4)
    grid.cells[2].source = LOCALIZADOR + "\n\n" + codigo_modulo("pipeline.py", sem_main=False) + "\n\nresultado = main()"
    salvar(LEGADO / "oat3_gridsearch.ipynb", grid)

    # Nomes antigos já compartilhados pela equipe: mesmo experimento de oat3_completa.
    for nome, titulo in [
        ("oat3_fechamento.ipynb", "MecaniQA - fechamento da OAT 3"),
        ("OAT_3_Tunando_Modelo_MecaniQA.ipynb", "MecaniQA - tunando o modelo"),
    ]:
        nb = copy.deepcopy(completa)
        nb.cells[0].source = (
            f"# {titulo}\n\nVersão mantida para links já compartilhados; executa o mesmo experimento de "
            "notebooks/oat3_completa.ipynb.\n\n" + "\n".join(completa.cells[0].source.splitlines()[2:])
        )
        salvar(LEGADO / nome, nb)
    return comum


def atualizar_oat2(comum):
    baselines = codigo_modulo("baseline_metricas.py")
    for caminho, titulo in [
        (NOTEBOOKS / "metricas_timeseries.ipynb", "MecaniQA - OAT 2: métricas e atributos temporais"),
        (LEGADO / "baseline_metricas.ipynb", "MecaniQA - baselines"),
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
        salvar(caminho, nb)


def atualizar_oat1():
    """Troca só o localizador da planilha no notebook da OAT 1, preservando a análise."""
    caminho = NOTEBOOKS / "mecaniqa_oat1.ipynb"
    nb = nbformat.read(caminho, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and LOCALIZADOR_ANTIGO_OAT1 in cell.source:
            inicio = cell.source.index("def localizar_planilha():")
            # "\n    return caminho" casa só com o return final (4 espaços), não com o do laço.
            fim = cell.source.index("\n    return caminho\n", inicio) + len("\n    return caminho\n")
            novo = LOCALIZADOR.split("\n\n\n", 1)[1]  # sem o "from pathlib import Path" inicial
            cell.source = cell.source[:inicio] + novo + cell.source[fim:]
            nbformat.write(nb, caminho)
            print("Atualizado (localizador):", caminho.relative_to(ROOT))
            return
    print("OAT 1 já usa o localizador atual.")


def main():
    comum = atualizar_oat3()
    atualizar_oat2(comum)
    atualizar_oat1()


if __name__ == "__main__":
    main()
