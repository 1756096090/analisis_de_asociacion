import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt

store_data = pd.read_csv('store_data.csv', header=None)

transacciones = []
for i in range(0, len(store_data)):
    transacciones.append([str(item) for item in store_data.iloc[i] if pd.notna(item)])

reglas_asociacion = apriori(
    transacciones,
    min_support=0.0045,
    min_confidence=0.2,
    min_lift=3,
    min_length=2
)

resultados = list(reglas_asociacion)


support_vals = []
confidence_vals = []
lift_vals = []
reglas = []

for regla in resultados:
    if len(regla.items) >= 2:
        soporte = regla.support
        confianza = regla.ordered_statistics[0].confidence
        lift = regla.ordered_statistics[0].lift
        support_vals.append(soporte)
        confidence_vals.append(confianza)
        lift_vals.append(lift)
        reglas.append(f"{list(regla.items)[0]} -> {list(regla.items)[1]}")


