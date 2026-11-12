#!/usr/bin/env python3
"""
Rulează pe rând notebook-urile din acest folder (00, 01, 02, ... 20) și scrie rezultatele în loguri/.

    python ruleaza_toate.py                  # toate, sare peste cele terminate deja
    python ruleaza_toate.py --doar 03,14     # doar anumite experimente
    python ruleaza_toate.py --de-la 07       # de la 07 încolo
    python ruleaza_toate.py --reia-tot       # le rulează din nou și pe cele terminate
    python ruleaza_toate.py --clasament      # doar refă clasamentul din ce există

Ce scrie în loguri/:
    ruleaza_toate.log         – ce a făcut scriptul (pornit, terminat, erori), cu ora
    rezumat.log               – după fiecare notebook: metricile lui (se adaugă la final)
    clasament.txt             – toate experimentele, ordonate după scorul pe validare
    rezultate.csv             – același clasament, de deschis în Excel
    <experiment>.log          – tot ce a afișat notebook-ul (celulă cu celulă)
    <experiment>_antrenare.log – rezultatele fiecărei epoci, scrise în timp ce antrenează
    <experiment>.json         – metricile complete (le scrie notebook-ul)
Notebook-urile rulate (cu grafice) ajung în executate/.
"""
import argparse
import csv
import datetime as dt
import importlib
import json
import re
import sys
import time
from pathlib import Path

BAZA = Path(__file__).resolve().parent
LOGURI = BAZA / "loguri"
EXECUTATE = BAZA / "executate"
DATE_GATA = BAZA / "date" / "dataset_seg" / "data_seg.yaml"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SCOR = ("validare", "mask", "mAP50-95")     # după ce ordonăm clasamentul


def acum():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def jurnal(mesaj):
    linie = f"[{acum()}] {mesaj}"
    print(linie, flush=True)
    with open(LOGURI / "ruleaza_toate.log", "a", encoding="utf-8") as f:
        f.write(linie + "\n")


def verifica_pachete():
    lipsa = []
    for modul, pachet in [("nbclient", "nbclient"), ("nbformat", "nbformat"), ("ipykernel", "ipykernel"),
                          ("ultralytics", "ultralytics"), ("torch", "torch"), ("cv2", "opencv-python"),
                          ("albumentations", "albumentations"), ("pandas", "pandas"), ("yaml", "pyyaml"),
                          ("matplotlib", "matplotlib")]:
        try:
            importlib.import_module(modul)
        except Exception:
            lipsa.append(pachet)
    if lipsa:
        print("❌ Lipsesc pachetele:", " ".join(lipsa))
        print("   Instalează-le cu:  pip install -r requirements.txt")
        sys.exit(1)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NICIUN GPU (va fi foarte lent)"
    import ultralytics
    jurnal(f"Python {sys.version.split()[0]} | ultralytics {ultralytics.__version__} | GPU: {gpu}")


# ----------------------------------------------------------------------------- rulare notebook
def text_iesiri(cell):
    bucati = []
    for out in cell.get("outputs", []):
        tip = out.get("output_type")
        if tip == "stream":
            bucati.append(out.get("text", ""))
        elif tip == "error":
            bucati.append("\n".join(out.get("traceback", [])) + "\n")
        elif tip in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data and not any(k.startswith("image/") for k in data):
                bucati.append(data["text/plain"] + "\n")
            elif any(k.startswith("image/") for k in data):
                bucati.append("[imagine – vezi notebook-ul din executate/]\n")
    text = ANSI.sub("", "".join(bucati))
    # barele de progres (\r) – păstrăm doar ultima stare a fiecărei linii
    return "\n".join(l.split("\r")[-1] for l in text.split("\n"))


def ruleaza_notebook(cale, kernel):
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(cale, as_version=4)
    log = open(LOGURI / f"{cale.stem}.log", "a", encoding="utf-8")
    log.write(f"\n\n========== {cale.name} – pornit la {acum()} ==========\n")
    log.flush()
    scrise = set()

    def la_start(cell, cell_index, **_):
        if cell.cell_type == "code":
            log.write(f"\n--- celula {cell_index} (pornită la {acum()}) ---\n")
            log.flush()

    def dupa_celula(cell, cell_index, **_):
        if cell_index not in scrise:
            scrise.add(cell_index)
            log.write(text_iesiri(cell))
            log.flush()

    client = NotebookClient(nb, timeout=None, kernel_name=kernel, allow_errors=False,
                            resources={"metadata": {"path": str(BAZA)}},
                            on_cell_start=la_start, on_cell_executed=dupa_celula)
    eroare = None
    try:
        client.execute()
    except CellExecutionError as e:
        eroare = ANSI.sub("", str(e)).strip().splitlines()
        eroare = " | ".join(l for l in eroare[-6:] if l.strip())[-1500:]
    except KeyboardInterrupt:
        eroare = "oprit de utilizator (Ctrl+C)"
        raise
    except Exception as e:
        eroare = f"{type(e).__name__}: {e}"
    finally:
        for i, cell in enumerate(nb.cells):          # rezervă: dacă hook-urile n-au mers (nbclient vechi)
            if cell.cell_type == "code" and i not in scrise and cell.get("outputs"):
                log.write(f"\n--- celula {i} ---\n" + text_iesiri(cell))
        log.write(f"\n========== {cale.name} – terminat la {acum()} – {'EROARE: ' + eroare if eroare else 'OK'} ==========\n")
        log.close()
        EXECUTATE.mkdir(exist_ok=True)
        nbformat.write(nb, EXECUTATE / cale.name)
    return eroare


# ----------------------------------------------------------------------------- rezultate
def citeste(stem):
    f = LOGURI / f"{stem}.json"
    try:
        return json.load(open(f, encoding="utf-8")) if f.exists() else None
    except Exception:
        return None


def scrie_json(stem, date):
    json.dump(date, open(LOGURI / f"{stem}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def metrica(r, set_, tip, cheie):
    try:
        return r["metrici"][set_][tip][cheie]
    except (KeyError, TypeError):
        return None


def bloc_rezumat(r):
    linii = [f"\n=== {r['experiment']} | {r.get('stare')} | {r.get('terminat_la', acum())} | "
             f"{r.get('durata_min', '?')} min | epoci {r.get('epoci_rulate', '?')}/{r.get('epoci_maxime', '?')} ==="]
    if r.get("descriere"):
        linii.append(f"    {r['descriere']}")
    if r.get("stare") != "gata":
        linii.append(f"    EROARE: {r.get('eroare', '?')}")
        return "\n".join(linii) + "\n"
    for set_, m in r.get("metrici", {}).items():
        for tip, v in m.items():
            linii.append(f"    {set_:13s} {tip:4s}  mAP50={v['mAP50']:.3f}  mAP50-95={v['mAP50-95']:.3f}  "
                         f"P={v['P']:.3f}  R={v['R']:.3f}")
    clase = metrica(r, "validare", "mask", "mAP50_pe_clase") or metrica(r, "validare", "box", "mAP50_pe_clase") or {}
    if clase:
        linii.append("    validare pe clase (mAP50): " + ", ".join(f"{k}={v:.3f}" for k, v in clase.items()))
    return "\n".join(linii) + "\n"


def refa_clasament(notebook_uri):
    randuri, clase = [], []
    for nb in notebook_uri:
        if nb.stem.startswith("00_"):
            continue
        r = citeste(nb.stem) or {"experiment": nb.stem, "stare": "nerulat"}
        rand = {"experiment": nb.stem, "stare": r.get("stare"), "durata_min": r.get("durata_min"),
                "epoci": r.get("epoci_rulate")}
        for set_ in ("validare", "test_complet", "test_vechi", "test_nou"):
            for tip in ("box", "mask"):
                for k in ("mAP50", "mAP50-95"):
                    rand[f"{set_}_{tip}_{k}"] = metrica(r, set_, tip, k)
        pe_clase = metrica(r, "validare", "mask", "mAP50_pe_clase") or {}
        for c, v in pe_clase.items():
            rand[f"validare_mask_{c}"] = v
            if c not in clase:
                clase.append(c)
        rand["descriere"] = r.get("descriere", "")
        rand["eroare"] = r.get("eroare", "")
        randuri.append(rand)

    cheie = f"{SCOR[0]}_{SCOR[1]}_{SCOR[2]}"
    randuri.sort(key=lambda x: (x[cheie] is None, -(x[cheie] or 0)))
    coloane = list(dict.fromkeys(k for r in randuri for k in r))
    with open(LOGURI / "rezultate.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=coloane)
        w.writeheader()
        w.writerows(randuri)

    def n(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "

    cap = (f"{'#':>2}  {'experiment':40s} {'stare':8s} {'VAL mask':>8s} {'VAL mask':>8s} {'VAL box':>8s} "
           f"{'TEST mask':>9s} {'TEST box':>8s} {'min':>6s}\n"
           f"{'':>2}  {'':40s} {'':8s} {'50-95':>8s} {'50':>8s} {'50':>8s} {'50-95':>9s} {'50':>8s}\n")
    linii = [f"CLASAMENT – actualizat la {acum()}",
             "Ordonat după validare Mask mAP50-95 (după asta alegi câștigătorul; testul doar se raportează).", "", cap]
    for i, r in enumerate(randuri, 1):
        linii.append(f"{i:>2}  {r['experiment']:40s} {str(r['stare']):8s} {n(r['validare_mask_mAP50-95']):>8s} "
                     f"{n(r['validare_mask_mAP50']):>8s} {n(r['validare_box_mAP50']):>8s} "
                     f"{n(r['test_complet_mask_mAP50-95']):>9s} {n(r['test_complet_box_mAP50']):>8s} "
                     f"{(str(round(r['durata_min'])) if isinstance(r['durata_min'], (int, float)) else '-'):>6s}")
    if clase:
        linii += ["", "Validare Mask mAP50 pe clase:", f"{'':2}  {'experiment':40s} " +
                  " ".join(f"{c[:10]:>10s}" for c in clase)]
        for i, r in enumerate(randuri, 1):
            linii.append(f"{i:>2}  {r['experiment']:40s} " +
                         " ".join(f"{n(r.get(f'validare_mask_{c}')):>10s}" for c in clase))
    (LOGURI / "clasament.txt").write_text("\n".join(linii) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Rulează pe rând notebook-urile de experimente.")
    p.add_argument("--doar", help="ex. 03,14 – doar aceste experimente")
    p.add_argument("--de-la", help="ex. 07 – de la acest experiment încolo")
    p.add_argument("--reia-tot", action="store_true", help="rulează din nou și experimentele terminate")
    p.add_argument("--clasament", action="store_true", help="doar refă clasament.txt și rezultate.csv")
    p.add_argument("--kernel", default="python3", help="kernel-ul Jupyter (implicit python3)")
    a = p.parse_args()

    LOGURI.mkdir(exist_ok=True)
    notebook_uri = sorted(BAZA.glob("[0-9][0-9]_*.ipynb"))
    experimente = [nb for nb in notebook_uri if not nb.stem.startswith("00_")]
    if a.clasament:
        refa_clasament(notebook_uri)
        print((LOGURI / "clasament.txt").read_text(encoding="utf-8"))
        return

    if a.doar:
        coduri = {c.strip().zfill(2) for c in a.doar.split(",") if c.strip()}
        experimente = [nb for nb in experimente if nb.stem[:2] in coduri]
    if a.de_la:
        experimente = [nb for nb in experimente if nb.stem[:2] >= a.de_la.zfill(2)]

    verifica_pachete()
    jurnal(f"=== Pornesc: {len(experimente)} experimente: {', '.join(nb.stem[:2] for nb in experimente)} ===")

    # 00 – datele, o singură dată
    pregatire = BAZA / "00_pregatire_date.ipynb"
    if not DATE_GATA.exists():
        jurnal("Datele lipsesc → rulez 00_pregatire_date")
        t0 = time.time()
        eroare = ruleaza_notebook(pregatire, a.kernel)
        if eroare or not DATE_GATA.exists():
            jurnal(f"❌ 00_pregatire_date a eșuat: {eroare}. Fără date nu pot continua – vezi loguri/00_pregatire_date.log")
            sys.exit(1)
        jurnal(f"✅ 00_pregatire_date gata în {(time.time() - t0) / 60:.1f} min")
    else:
        jurnal("Datele există deja (date/dataset_seg) → sar peste 00")

    for i, nb in enumerate(experimente, 1):
        vechi = citeste(nb.stem)
        if vechi and vechi.get("stare") == "gata" and not a.reia_tot:
            jurnal(f"⏭️  [{i}/{len(experimente)}] {nb.stem} e deja gata – sar peste")
            continue
        dosar = BAZA / "runs" / nb.stem
        if a.reia_tot and dosar.exists():   # de la zero: antrenarea veche se păstrează, redenumită
            dosar.rename(dosar.with_name(f"{nb.stem}_vechi_{dt.datetime.now():%Y%m%d_%H%M}"))
        if (LOGURI / f"{nb.stem}.json").exists():
            (LOGURI / f"{nb.stem}.json").unlink()      # rezultatul vechi (eroare sau --reia-tot) se înlocuiește
        jurnal(f"▶️  [{i}/{len(experimente)}] pornesc {nb.stem}")
        t0 = time.time()
        try:
            eroare = ruleaza_notebook(nb, a.kernel)
        except KeyboardInterrupt:
            jurnal(f"⛔ Oprit de utilizator în timpul {nb.stem}. La repornire continuă de unde a rămas.")
            refa_clasament(notebook_uri)
            sys.exit(130)
        durata = round((time.time() - t0) / 60, 1)
        r = citeste(nb.stem)
        if eroare or not r or r.get("stare") != "gata":
            r = {"experiment": nb.stem, "stare": "eroare", "eroare": eroare or "notebook-ul nu a scris rezultatul",
                 "terminat_la": acum()}
            jurnal(f"❌ {nb.stem} a eșuat după {durata} min: {r['eroare'][:300]}")
        else:
            jurnal(f"✅ {nb.stem} gata în {durata} min | validare Mask mAP50-95 = "
                   f"{metrica(r, 'validare', 'mask', 'mAP50-95')} | Box mAP50 = {metrica(r, 'validare', 'box', 'mAP50')}")
        r["durata_min"] = durata
        scrie_json(nb.stem, r)
        with open(LOGURI / "rezumat.log", "a", encoding="utf-8") as f:
            f.write(bloc_rezumat(r))
        refa_clasament(notebook_uri)

    refa_clasament(notebook_uri)
    jurnal("=== Gata toate. Clasamentul: loguri/clasament.txt | tabel: loguri/rezultate.csv ===")
    print((LOGURI / "clasament.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
