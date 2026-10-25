"""Executa todos os notebooks em uma cópia temporária, com o Python atual."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
SAIDA = ROOT / "output" / "testes"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    resultados = []
    with tempfile.TemporaryDirectory(prefix="mecaniqa_notebooks_") as temporario:
        trabalho = Path(temporario)
        kernels = trabalho / "kernels"
        kernel = kernels / "mecaniqa_testes"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "MecaniQA testes",
            "language": "python",
        }), encoding="utf-8")
        spec = KernelSpecManager(kernel_dirs=[str(kernels)])
        for arquivo in sorted(ROOT.glob("*.ipynb")):
            pasta = trabalho / arquivo.stem
            (pasta / "datasets").mkdir(parents=True)
            shutil.copy2(ROOT / "datasets/mecaniqa_dataset.xlsx", pasta / "datasets/mecaniqa_dataset.xlsx")
            nb = nbformat.read(arquivo, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            inicio = perf_counter()
            resultado = {"arquivo": arquivo.name}
            print(f"Executando {arquivo.name}", flush=True)
            km = KernelManager(kernel_name="mecaniqa_testes", kernel_spec_manager=spec)
            cliente = NotebookClient(nb, km=km, timeout=180, resources={"metadata": {"path": str(pasta)}})
            try:
                cliente.execute(cwd=str(pasta))
                resultado["status"] = "OK"
            except Exception as erro:
                resultado["status"] = "ERRO"
                resultado["erro"] = str(erro)
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
            resultado["segundos"] = round(perf_counter() - inicio, 2)
            resultado["graficos"] = sum(
                "image/png" in output.get("data", {})
                for cell in nb.cells for output in cell.get("outputs", [])
            )
            resultado["texto"] = "\n".join(
                output.get("text", "")
                for cell in nb.cells for output in cell.get("outputs", [])
                if output.output_type == "stream"
            )
            nbformat.write(nb, SAIDA / arquivo.name)
            if (pasta / "output/oat3").exists():
                shutil.copytree(pasta / "output/oat3", SAIDA / arquivo.stem, dirs_exist_ok=True)
            resultados.append(resultado)
            (SAIDA / "notebooks.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{resultado['status']}: {arquivo.name} ({resultado['segundos']} s)", flush=True)
    return 0 if all(r["status"] == "OK" for r in resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
