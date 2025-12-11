# Watercraft-detection


## Zbiory danych

Projekt korzysta z pkilku zbiorów danych dotyczących statków i jednostek pływających. Wszystkie zbiory, pierwotnie w formatach COCO, YOLO oraz XML (Pascal VOC), zostały zunifikowane do formatu COCO, co umożliwia ich bezpośrednie wykorzystanie w zadaniach detekcji obiektów.  

[LINK Google Drive](https://drive.google.com/file/d/1HUypJeqkcTtgJBapiag6fEMVE9HuG_wB/view?usp=drive_link)

### Szczegóły zbiorów

 Wszystkie zbiory zostały dostosowane do **formatu COCO**, co zapewnia spójność adnotacji i gotowość do trenowania modeli detekcji obiektów.

| Dataset | Link | Licencja / Prawa | Train | Valid | Test | Łącznie | Preprocessing | Augmentacje |
|---------|------|-----------------|------:|------:|-----:|--------:|---------------|-------------|
| navy ships Computer Vision Model | [LINK](https://universe.roboflow.com/new-workspace-kgvwb/navy-ships-dywua/dataset/9) | CC BY 4.0 | 2259 | 223 | 104 | 2586 | Auto-Orient, Resize 416x416 | Rotation ±10°, Brightness ±20%, 3 wersje obrazu |
| Ships Computer Vision Dataset | [LINK](https://universe.roboflow.com/test-burb2/ships-ebt8m/dataset/1) | CC BY 4.0 | 5294 | 1671 | 927 | 7892 | Auto-Orient, Resize 640x640 | - |
| Multiclass_detection Computer Vision Model | [LINK](https://universe.roboflow.com/maritimeclassification/multiclass_detection) | CC BY 4.0 | 19746 | 1354 | 680 | 21780 | Auto-Orient, Resize 640x640 | Noise: Up to 0.14% of pixels |
| FastRCNN-Update Computer Vision Datase | [LINK](https://universe.roboflow.com/convert-g59qi/fastrcnn-update) | CC BY 4.0 | 4906 | 800 | 400 | 6106 | Auto-Orient, Resize 640x640 | Horizontal Flip, 3 wersje obrazu |
| Mcships | [LINK](https://github.com/ZhengYitong2333/Mcships/blob/master) | MIT License | 4048 | 1637 | 2311 | 7996 | - | Pascal VOC |
| Ships Image Dataset | [LINK](https://www.kaggle.com/datasets/vinayakshanawad/ships-dataset?resource=download) | Attribution 4.0 International | 7406 | 689 | 381 | 8476 | Auto-Orient, Resize to 600x410 | 50% probability of horizontal flip; Random Gaussian blur between 0 and 3.75 pixels |



## Statystyki zbiorów
Łącznie w zbiorach ogólnych znajduje się **54 463** obrazy. Podział na zbiory:  

| Split | Liczba zdjęć | Procent |
|-------|----------------:|--------:|
| Train | 43 347 | 79,59% |
| Valid | 6 339  | 11,64% |
| Test  | 4 777  | 8,77%  |
---

Łącznie w zbiorach użytych do treningu znajduje się **31 209** obrazy. Podział na zbiory:  

| Split | Liczba zdjęć | Procent |
|-------|----------------:|--------:|
| Train | 26 140 | 83,76% |
| Valid | 1 763  | 10,59% |
| Test  | 31 209  | 5,65%  |
---
