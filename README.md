# Experimente cu parametrii de augmentare (segmentare din .txt, GrabCut)

20 de notebook-uri identice cu `antrenare_yolo_segmentare_din_txt`. Singura diferență între ele este **combinația de parametri de augmentare**: `hsv_h`, `hsv_s`, `hsv_v`, `translate`, `scale`, `fliplr`, `mosaic`, `erasing`, `auto_augment` și transformările Albumentations `Blur` / `CLAHE`.
Scriptul `ruleaza_toate.py` le rulează pe rând și, după fiecare, scrie rezultatul în `loguri/`.

## Ce conține repo-ul

- `00_pregatire_date.ipynb`, cu datele (o singură dată), și `01_…` – `20_…`, cele 20 de combinații
- `ruleaza_toate.py`, care le rulează pe rând
- `imparte_arhiva.py`, care împarte arhiva cu date în bucăți pentru GitHub
- `yolo11m-seg.pt` (modelul de pornire) și `Arial.ttf` (fontul pentru grafice), ca să meargă fără internet
- (doar la varianta B de mai jos) `dataset_segmentare_grabcut_cu_date_noi.zip.part01 … part04`: datele, adică pozele plus poligoanele GrabCut, împărțite în bucăți

## Arhiva cu date (o singură dată)

**Varianta A – arhiva ajunge separat pe placă** (stick, link Google Drive, WeTransfer etc.): o pui așa cum e, **nedezarhivată**, în folderul `experimente_parametri` de pe placă. Gata, nu trebuie împărțită.

**Varianta B – arhiva intră în repo:** are ~300 MB, iar GitHub nu acceptă fișiere mai mari de 100 MB. De aceea:
1. Descarci `dataset_segmentare_grabcut_cu_date_noi.zip` din Google Drive → `licenta`. **Nu o dezarhiva.**
2. O pui în acest folder și rulezi `python imparte_arhiva.py`. Rezultă `….zip.part01`, `.part02` etc., de câte 90 MB.
3. `git add .` → bucățile intră în repo. Arhiva mare e în `.gitignore`, deci nu intră.

Pe placă, `00_pregatire_date` lipește singur bucățile și le dezarhivează în `date/dataset_seg/`.

## Pe placă

```bash
git clone <repo>  &&  cd experimente_parametri
pip install -r requirements.txt     # o singură dată
python ruleaza_toate.py
```

Pe un server la care te conectezi de la distanță, pornește-l așa încât să nu se oprească atunci când închizi fereastra:
```bash
nohup python ruleaza_toate.py > loguri/consola.txt 2>&1 &
tail -f loguri/ruleaza_toate.log
```

Alte variante:
```bash
python ruleaza_toate.py --doar 03,14    # doar anumite experimente
python ruleaza_toate.py --de-la 07      # de la 07 încolo
python ruleaza_toate.py --reia-tot      # din nou, inclusiv cele terminate
python ruleaza_toate.py --clasament     # doar afișează clasamentul
```

**Dacă placa se oprește:** pornești din nou aceeași comandă. Experimentele terminate se sar, iar cel întrerupt continuă de la ultima epocă.
**Dacă un experiment crapă**, scriptul notează eroarea și trece la următorul.

## Ce găsești în `loguri/`

| fișier | ce conține |
|---|---|
| `clasament.txt` | **aici te uiți primul**: toate experimentele ordonate după scor |
| `rezultate.csv` | același clasament, cu toate metricile; se deschide în Excel |
| `rezumat.log` | după fiecare notebook se adaugă un bloc cu metricile lui (validare, test vechi/nou/complet, pe clase) |
| `ruleaza_toate.log` | jurnalul scriptului: ce a pornit, când s-a terminat, erori |
| `NN_nume.log` | tot ce a afișat notebook-ul, celulă cu celulă |
| `NN_nume_antrenare.log` | rezultatele fiecărei epoci, scrise **în timp ce** antrenează |
| `NN_nume.json` / `NN_nume_metrici.csv` | metricile complete ale unui experiment |
| `00_verificare_contururi.png` | poze cu poligoanele, ca să verifici că GrabCut a mers |

`loguri/` nu e în `.gitignore`: după rulare, faci `git add loguri && git commit && git push` de pe placă și vezi rezultatele la tine.

## Cum citești scorul

- **Mask mAP50-95**: cât de bine se potrivește conturul (mai strict). După el se face clasamentul.
- **Box mAP50**: cât de bine găsește obiectul. Se poate compara cu exp2/exp3 (detecție).
- Câștigătorul se alege după **validare**. Testul îl raportezi în lucrare, nu alegi după el.

## Cele 20 de combinații

Comun tuturor: `yolo11m-seg`, 150 de epoci, patience 30, imgsz 640, batch 8, seed 0, aceleași date.
`erasing = 0` și `auto_augment = None` la toate: Ultralytics le folosește doar la clasificare, deci la segmentare nu schimbă nimic.
Valorile implicite Ultralytics sunt: hsv 0.015 / 0.7 / 0.4, translate 0.1, scale 0.5, fliplr 0.5, degrees 0, flipud 0, mosaic 1.0.
Notebook-ul original (`antrenare_yolo_segmentare_din_txt`) folosea implicit + degrees 15 + flipud 0.5. Asta e combinația 05.

| # | hsv_h | hsv_s | hsv_v | translate | scale | fliplr | degrees | flipud | mosaic | Albumentations |
|---|---|---|---|---|---|---|---|---|---|---|
| 01_toate_oprite | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | – |
| 02_toate_oprite_blur_clahe | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | Blur 7 + CLAHE 4 (p 0.5) |
| 03_implicit | 0.015 | 0.7 | 0.4 | 0.1 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | – |
| 04_implicit_blur_clahe | 0.015 | 0.7 | 0.4 | 0.1 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | Blur 7 + CLAHE 4 (p 0.5) |
| 05_implicit_rotire | 0.015 | 0.7 | 0.4 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 1.0 | – |
| 06_implicit_rotire_blur_clahe | 0.015 | 0.7 | 0.4 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 1.0 | Blur 7 + CLAHE 4 (p 0.5) |
| 07_doar_culoare | 0.015 | 0.7 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | – |
| 08_doar_culoare_blur_clahe | 0.015 | 0.7 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | Blur 7 + CLAHE 4 (p 0.5) |
| 09_doar_geometrie | 0.0 | 0.0 | 0.0 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 0.0 | – |
| 10_doar_geometrie_blur_clahe | 0.0 | 0.0 | 0.0 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 0.0 | Blur 7 + CLAHE 4 (p 0.5) |
| 11_doar_rotire | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15.0 | 0.5 | 0.0 | – |
| 12_implicit_rotire_fara_mosaic | 0.015 | 0.7 | 0.4 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 0.0 | – |
| 13_slab | 0.0075 | 0.35 | 0.2 | 0.05 | 0.25 | 0.5 | 5.0 | 0.25 | 1.0 | – |
| 14_slab_blur_clahe_slab | 0.0075 | 0.35 | 0.2 | 0.05 | 0.25 | 0.5 | 5.0 | 0.25 | 1.0 | Blur 3 + CLAHE 2 (p 0.2) |
| 15_puternic | 0.03 | 0.9 | 0.6 | 0.2 | 0.8 | 0.5 | 30.0 | 0.5 | 1.0 | – |
| 16_puternic_blur_clahe | 0.03 | 0.9 | 0.6 | 0.2 | 0.8 | 0.5 | 30.0 | 0.5 | 1.0 | Blur 7 + CLAHE 4 (p 0.5) |
| 17_implicit_rotire_doar_blur | 0.015 | 0.7 | 0.4 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 1.0 | doar Blur 7 (p 0.5) |
| 18_implicit_rotire_doar_clahe | 0.015 | 0.7 | 0.4 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 1.0 | doar CLAHE 4 (p 0.5) |
| 19_culoare_puternica_rotire_blur_clahe | 0.03 | 0.9 | 0.6 | 0.1 | 0.5 | 0.5 | 15.0 | 0.5 | 1.0 | Blur 7 + CLAHE 4 (p 0.5) |
| 20_geometrie_puternica_blur_clahe | 0.015 | 0.7 | 0.4 | 0.2 | 0.8 | 0.5 | 30.0 | 0.5 | 1.0 | Blur 7 + CLAHE 4 (p 0.5) |

## Cum adaugi al 21-lea

Copiezi orice notebook, de exemplu `05_implicit_rotire.ipynb` → `21_ce_vreau_eu.ipynb`.
În **prima celulă de cod** schimbi `NUME_EXPERIMENT = "21_ce_vreau_eu"`, `DESCRIERE`, `PARAMETRI` și `AUGMENTARI`.
Apoi rulezi `python ruleaza_toate.py --doar 21`.

## Cheia Roboflow

Nu e scrisă nicăieri în repo. Trebuie doar dacă placa trebuie să descarce singură datele, adică dacă lipsesc bucățile arhivei.
În cazul ăsta o pui în `cheie_roboflow.txt` (e în `.gitignore`) sau în variabila `ROBOFLOW_API_KEY`.
