# Watercraft-detection


## Zbiory danych

Projekt korzysta z pkilku zbiorów danych dotyczących statków i jednostek pływających. Wszystkie zbiory, pierwotnie w formatach COCO, YOLO oraz XML (Pascal VOC), zostały zunifikowane do formatu COCO, co umożliwia ich bezpośrednie wykorzystanie w zadaniach detekcji obiektów.  

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
Łącznie w zbiorach znajduje się **76 823 obiekty**.
Łącznie w zbiorach znajduje się **54 463** obrazy. Podział na zbiory:  

| Split | Liczba zdjęć | Procent |
|-------|----------------:|--------:|
| Train | 43 347 | 79,59% |
| Valid | 6 339  | 11,64% |
| Test  | 4 777  | 8,77%  |

---

## Wykryte zunifikowane klasy

Wszystkie klasy zostały zidentyfikowane i ujednolicone, co umożliwia spójne trenowanie modeli detekcji obiektów pływających.

| Unified Class      | Test | Train | Valid | Total |
|------------------|-----:|------:|-----:|------:|
| Aircraft carrier  | 184  | 2224  | 333  | 2741 |
| Boat              | 303  | 2354  | 811  | 3468 |
| Bulkers           | 11   | 200   | 19   | 230  |
| Buoy              | 104  | 3612  | 225  | 3941 |
| Canoe             | 432  | 2148  | 706  | 3286 |
| Car carrier       | 90   | 1772  | 179  | 2041 |
| Civilian ship     | 1743 | 3010  | 1196 | 5949 |
| Coastguard ship   | 26   | 236   | 66   | 328  |
| Container ship    | 397  | 6124  | 704  | 7225 |
| Cruise            | 535  | 9530  | 1047 | 11112|
| Destroyer         | 94   | 2394  | 225  | 2713 |
| Fishing boat      | 355  | 7211  | 612  | 8178 |
| Human             | 183  | 1333  | 407  | 1923 |
| Landing ship      | 54   | 1229  | 107  | 1390 |
| Recreational      | 16   | 236   | 11   | 263  |
| Sail boat         | 435  | 4627  | 916  | 5978 |
| Submarine         | 95   | 1542  | 215  | 1852 |
| Tanker            | 82   | 1029  | 157  | 1268 |
| Tug               | 35   | 803   | 77   | 915  |
| Warship           | 1823 | 8454  | 1745 | 12022|


