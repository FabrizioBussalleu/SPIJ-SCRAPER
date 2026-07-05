"""Setup local del proyecto SPIJ-SCRAPER.

Ejecuta en orden: dependencias, BD, datos, indice FAISS, modelos y assets de presentacion.
No modifica la logica del scraper ni de los modelos; solo orquesta lo existente.

Uso:
    py -3 setup_local.py                  # pipeline completo (import JSON + index + report)
    py -3 setup_local.py --scrape 200     # scrapear 200 normas del SPIJ en lugar de JSON
    py -3 setup_local.py --skip-shap       # omitir SHAP (mas rapido)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list, desc: str, timeout: int = None) -> bool:
    print(f"\n{'='*60}\n  {desc}\n  > {' '.join(cmd)}\n{'='*60}")
    try:
        result = subprocess.run(cmd, cwd=ROOT, timeout=timeout)
        if result.returncode != 0:
            print(f"  [AVISO] Termino con codigo {result.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [AVISO] Timeout tras {timeout}s")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Setup local SPIJ-SCRAPER")
    p.add_argument("--scrape", type=int, default=0, help="Normas a scrapear del SPIJ (0= usar JSON)")
    p.add_argument("--skip-shap", action="store_true", help="Omitir SHAP en presentation_report")
    p.add_argument("--skip-index", action="store_true", help="Omitir embeddings_indexer")
    args = p.parse_args()

    print("Setup SPIJ-SCRAPER — inicio")

    # 1. Dependencias
    run([PY, "-m", "pip", "install", "-r", "requirements.txt", "-q"], "Instalando dependencias")

    # 2. Base de datos
    run([PY, "main.py", "init-db"], "Inicializando base de datos")
    run([PY, "main.py", "migrate-db"], "Migraciones")

    # 3. Datos
    if args.scrape > 0:
        ok = run(
            [PY, "main.py", "scrape", "--limit", str(args.scrape), "--validate-all", "--rate-limit", "1.5"],
            f"Scraping {args.scrape} normas del SPIJ (puede tardar varios minutos)",
            timeout=3600,
        )
        if not ok:
            print("  Scrape fallo; intentando import JSON como respaldo...")
            run([PY, "import_json.py"], "Importando JSON de respaldo")
    else:
        run([PY, "import_json.py"], "Importando corpus desde ai-training-dataset/normas_export.json")

    # 4. Indice FAISS
    if not args.skip_index:
        run(
            [PY, "embeddings_indexer.py", "--limit", "0"],
            "Generando embeddings e indice FAISS (descarga modelo ~1.1 GB la primera vez)",
            timeout=7200,
        )

    # 5. Assets de presentacion
    cmd = [PY, "presentation_report.py", "--k", "5"]
    if not args.skip_shap:
        cmd.append("--shap")
    run(cmd, "Generando graficos y guia de presentacion", timeout=3600)

    print("\n" + "=" * 60)
    print("  SETUP COMPLETADO")
    print(f"  Graficos:  {ROOT / 'exports' / 'presentacion'}")
    print(f"  Guia:      {ROOT / 'docs' / 'GUIA_PRESENTACION.md'}")
    print(f"  BD:        {ROOT / 'data' / 'norms.db'}")
    print("=" * 60)
    print("\nPara corpus completo (~1035 normas, ~22 min):")
    print("  py -3 main.py scrape --validate-all --rate-limit 1.5")
    print("  py -3 embeddings_indexer.py --rebuild")
    print("  py -3 presentation_report.py --k 5 --shap")


if __name__ == "__main__":
    main()
