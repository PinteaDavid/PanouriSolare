#!/usr/bin/env python3
"""
Împarte arhiva cu date în bucăți de 90 MB, ca să încapă în GitHub (limita e 100 MB pe fișier).

    python imparte_arhiva.py

Caută în acest folder dataset_segmentare_*.zip și scrie lângă ea:
    dataset_segmentare_grabcut_cu_date_noi.zip.part01, .part02, ...
Bucățile intră în repo; arhiva mare NU (e în .gitignore).
00_pregatire_date le lipește la loc singur.
"""
import glob
import hashlib
import os
import sys

MARIME = 90 * 1024 * 1024
BAZA = os.path.dirname(os.path.abspath(__file__))

arhive = sorted(glob.glob(os.path.join(BAZA, "dataset_segmentare_*.zip")))
if not arhive:
    sys.exit("Nu găsesc nicio arhivă dataset_segmentare_*.zip în folder. Descarc-o din Google Drive → licenta.")
for arhiva in arhive:
    for vechi in glob.glob(arhiva + ".part*"):
        os.remove(vechi)
    h_original, bucati = hashlib.md5(), []
    with open(arhiva, "rb") as f:
        i = 1
        while True:
            date = f.read(MARIME)
            if not date:
                break
            h_original.update(date)
            cale = f"{arhiva}.part{i:02d}"
            with open(cale, "wb") as out:
                out.write(date)
            bucati.append(cale)
            i += 1
    h_bucati = hashlib.md5()
    for b in bucati:
        h_bucati.update(open(b, "rb").read())
    assert h_bucati.hexdigest() == h_original.hexdigest(), "verificarea a eșuat"
    print(f"✅ {os.path.basename(arhiva)} → {len(bucati)} bucăți (verificat):")
    for b in bucati:
        print(f"   {os.path.basename(b)}  {os.path.getsize(b) / 1e6:.0f} MB")
print("Acum poți face git add . && git commit (arhiva mare rămâne doar la tine).")
