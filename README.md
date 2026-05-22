# Anomalieerkennung für Predictive Maintenance

Kursprojekt im Rahmen des Kurses **Grundlagen Data Science und KI** an der DHBW Mannheim.

Implementierung und Vergleich zweier unsupervised Anomalieerkennungsverfahren auf dem NASA C-MAPSS FD001 Datensatz für Predictive Maintenance von Turbofan-Triebwerken.

## Modelle

- **Isolation Forest** mit Rolling-Mean-Glättung und unit-spezifischer Z-Score-Normalisierung
- **LSTM Autoencoder** mit Sliding-Window-Sequenzen und Rekonstruktionsfehler-basierter Anomalieerkennung

## Ergebnisse (Trainingsdatensatz, RUL-Schwellenwert = 30)

| Modell | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest (optimiert) | 0.579 | 0.844 | 0.687 |
| LSTM Autoencoder (optimiert) | 0.812 | 0.859 | 0.834 |

## Installation

```bash
git clone https://github.com/mohathecreator/anomalie_erkennung_projekt
cd anomalie_erkennung_projekt
pip install -r requirements.txt
```

## Verwendung

**Isolation Forest trainieren:**
```bash
python 01_src/iso_forest/iso_forest_train.py
```

**LSTM Autoencoder trainieren:**
```bash
python 01_src/autoencoder/autoencoder_train.py
```

**Evaluation auf Testdatensatz:**
```bash
python 01_src/iso_forest/iso_forest_test.py
python 01_src/autoencoder/autoencoder_test.py
```

**Grid Search (Hyperparameter-Tuning):**
```bash
python 01_src/iso_forest/iso_forest_gridsearch.py
python 01_src/autoencoder/autoencoder_gridsearch.py
```

## Datensatz

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001.
Verfügbar unter: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## Abhängigkeiten

- Python 3.14
- pandas
- numpy
- scikit-learn
- torch (PyTorch 2.10)
- matplotlib