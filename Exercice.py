import pandas as pd
import numpy as np


df = pd.read_csv("apple_stock.csv")


df.columns = df.columns.str.strip()

resultats = {}

for variable in ["Open", "Close"]:

    x = pd.to_numeric(df[variable], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) < 4:
        raise ValueError(
            f"{variable} : au moins 4 valeurs valides sont nécessaires."
        )

    #  Effectifs et fréquences par intervalles
    # Règle de Sturges pour choisir le nombre d'intervalles
    nombre_intervalles = int(np.ceil(1 + np.log2(len(x))))

    intervalles = pd.cut(
        x,
        bins=nombre_intervalles,
        include_lowest=True
    )

    effectifs = intervalles.value_counts(sort=False)

    tableau_frequences = pd.DataFrame({
        "Effectif": effectifs,
        "Fréquence": effectifs / len(x),
        "Pourcentage (%)": effectifs / len(x) * 100
    })

    # Toutes les classes modales en cas d'égalité
    classes_modales = effectifs[
        effectifs == effectifs.max()
    ].index

    # 3. Quartiles et IQR
    q1 = x.quantile(0.25)
    q2 = x.quantile(0.50)
    q3 = x.quantile(0.75)

    iqr = q3 - q1

    # 4. Bornes et comptage des valeurs atypiques
    borne_inferieure = q1 - 1.5 * iqr
    borne_superieure = q3 + 1.5 * iqr

    nb_inferieur = (x < borne_inferieure).sum()
    nb_superieur = (x > borne_superieure).sum()

    # 5. Toutes les mesures statistiques
    resultats[variable] = {
        "Effectif valide": len(x),
        "Valeurs exclues": len(df) - len(x),
        "Moyenne": x.mean(),
        "Médiane": x.median(),
        "Minimum": x.min(),
        "P5": x.quantile(0.05),
        "Q1": q1,
        "Q2 (médiane)": q2,
        "Q3": q3,
        "P95": x.quantile(0.95),
        "Maximum": x.max(),
        "Étendue": x.max() - x.min(),
        "Variance (division par n)": x.var(ddof=0),
        "Écart-type (division par n)": x.std(ddof=0),
        "IQR": iqr,
        "Borne inférieure": borne_inferieure,
        "Borne supérieure": borne_superieure,
        "Nombre sous la borne inférieure": nb_inferieur,
        "Nombre au-dessus de la borne supérieure": nb_superieur,
        "Nombre total de valeurs atypiques": nb_inferieur + nb_superieur,
        "Skewness": x.skew(),
        "Excès de kurtosis": x.kurt()
    }


    print(f"\n{'_' * 60}")
    print(f"VARIABLE : {variable}")
    print(f"{'_' * 60}")

    print("\n Effectifs et fréquences par intervalles de prix :")
    print(tableau_frequences.round(4).to_string())

    print("\nClasse(s) modale(s) :")
    for classe in classes_modales:
        print(f"{classe} : {effectifs.max()} observations")

# 7. Tableau récapitulatif Open / Close
tableau_statistiques = pd.DataFrame(resultats)

print("\nTABLEAU DES MESURES STATISTIQUES")
print(tableau_statistiques.round(4).to_string())



df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Essential: sort from oldest to newest
df = df.sort_values("Date").reset_index(drop=True)




# Compare Close with the previous session
previous_close = df["Close"].shift(1)
valid = df["Close"].notna() & previous_close.notna()

# Keep unknown directions missing, including the first row
df["Direction"] = pd.Series(pd.NA, index=df.index, dtype="string")

df.loc[valid & (df["Close"] > previous_close), "Direction"] = "Hausse"
df.loc[valid & (df["Close"] < previous_close), "Direction"] = "Baisse"
df.loc[valid & (df["Close"] == previous_close), "Direction"] = "Stable"

# Display the result
print(df[["Date", "Close", "Direction"]].head(10))

# Save a new CSV
df.to_csv("apple_stock_direction.csv", index=False)