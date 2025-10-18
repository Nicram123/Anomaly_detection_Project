# Anomaly_Detection
## Opis
Notebook służy do analizy danych i detekcji anomalii. Zawiera kroki związane z:
- wczytywaniem i przygotowaniem danych do analizy,
- eksploracją i wizualizacją,
- trenowaniem modeli (XGB, Regresja, RandomForest),
- ewaluacją wyników i wizualizacją rezultatów
## Przykładowe wykresy

## Wykres: Heatmap korelacji

<img width="850" height="772" alt="image" src="https://github.com/user-attachments/assets/5d71f6d0-7d03-43b9-9754-50e3431a2d4c" />

## Wykres: Rozkład transakcji / histogram

<img width="842" height="262" alt="image" src="https://github.com/user-attachments/assets/d1757d37-43d5-4392-a6ec-d79143a887a1" />

<img width="843" height="261" alt="image" src="https://github.com/user-attachments/assets/f22effd7-8cd8-434a-b1b2-a772707f153d" />

## Redukcja wymiarów 

<img width="835" height="237" alt="image" src="https://github.com/user-attachments/assets/65aba148-325b-4b40-a945-75868ddde6db" />

### Balansowanie klas (jeśli występuje nierównowaga) — przykład SMOTE 

```python from imblearn.over_sampling import SMOTE sm = SMOTE(random_state=42) X_res, y_res = sm.fit_resample(X_train, y_train) 
```

