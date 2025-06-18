import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ────────────────────────────────────────────────
# 1. Cargar el CSV
# ────────────────────────────────────────────────
df_raw = pd.read_csv('store_data.csv', header=None)

# Convertir filas a listas (transacciones)
transactions = df_raw.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# ────────────────────────────────────────────────
# 2. One-hot encoding con TransactionEncoder
# ────────────────────────────────────────────────
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_hot = pd.DataFrame(te_ary, columns=te.columns_)

# ────────────────────────────────────────────────
# 3. Minería de patrones con Apriori
# ────────────────────────────────────────────────
min_support = 0.004  # Puedes ajustar esto si quieres más reglas
frequent_itemsets = apriori(df_hot, min_support=min_support, use_colnames=True)

# ────────────────────────────────────────────────
# 4. Generar reglas de asociación
# ────────────────────────────────────────────────
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
rules = rules.sort_values(['confidence', 'lift'], ascending=False)

# Formato de decimales
pd.options.display.float_format = '{:.2f}'.format

# Mostrar todas las reglas encontradas
print("\n📘 Reglas de asociación generadas:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ────────────────────────────────────────────────
# 5. Función de recomendación personalizada
# ────────────────────────────────────────────────
def recomendar(productos_comprados, reglas, top_n=10):
    """
    Solo recomienda productos si los productos del usuario coinciden
    exactamente con el conjunto del antecedente de la regla.
    """
    candidatos = []

    for _, row in reglas.iterrows():
        antecedente = row['antecedents']
        if antecedente == productos_comprados and len(row['consequents']) == 1:
            item = next(iter(row['consequents']))
            candidatos.append({
                'Regla usada': f"{', '.join(antecedente)} ⇒ {item}",
                'Producto recomendado': item,
                'Confianza': row['confidence'],
                'Soporte': row['support'],
                'Lift': row['lift']
            })

    # Ordenar por confianza y limitar
    recomendaciones = sorted(candidatos, key=lambda x: x['Confianza'], reverse=True)[:top_n]
    return recomendaciones

# ────────────────────────────────────────────────
# 6. Recomendaciones para el usuario
# ────────────────────────────────────────────────
productos_usuario = {'eggs', 'spaghetti', 'mineral water'}
sugerencias = recomendar(productos_usuario, rules, top_n=10)

# Convertir resultados a DataFrame y mostrar
df_resultado = pd.DataFrame(sugerencias)

print("\n📦 Productos del usuario:", productos_usuario)
if not df_resultado.empty:
    print("\n✅ Recomendaciones en tabla con reglas aplicadas:\n")
    print(df_resultado[['Regla usada', 'Producto recomendado', 'Confianza', 'Soporte', 'Lift']])
else:
    print("\n⚠️ No se encontraron recomendaciones exactas para estos productos.")
