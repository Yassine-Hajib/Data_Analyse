
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# ==========================================

df = pd.read_csv("apple_stock.csv")

df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df = df.sort_values("Date").reset_index(drop=True)

resultats = {}


# ==========================================
# 2. ANALYSE DES VARIABLES OPEN ET CLOSE
# ==========================================

for variable in ["Open", "Close"]:

    x = pd.to_numeric(df[variable], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) < 4:
        raise ValueError(
            f"{variable} : au moins 4 valeurs valides sont nécessaires."
        )

    # Effectifs et frequences par intervalles
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

    classes_modales = effectifs[
        effectifs == effectifs.max()
    ].index

    # Quartiles et IQR
    q1 = x.quantile(0.25)
    q2 = x.quantile(0.50)
    q3 = x.quantile(0.75)

    iqr = q3 - q1

    # Bornes et valeurs atypiques
    borne_inferieure = q1 - 1.5 * iqr
    borne_superieure = q3 + 1.5 * iqr

    nb_inferieur = (x < borne_inferieure).sum()
    nb_superieur = (x > borne_superieure).sum()

    # Mesures statistiques
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

    print("\nEffectifs et fréquences par intervalles de prix :")
    print(tableau_frequences.round(4).to_string())

    print("\nClasse(s) modale(s) :")

    for classe in classes_modales:
        print(f"{classe} : {effectifs.max()} observations")


# ==========================================
# 3. TABLEAU RECAPITULATIF OPEN / CLOSE
# ==========================================

tableau_statistiques = pd.DataFrame(resultats)

print("\nTABLEAU DES MESURES STATISTIQUES")
print(tableau_statistiques.round(4).to_string())


# ==========================================
# 4. CREATION DE LA VARIABLE DIRECTION
# ==========================================

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Comparer le Close du jour avec celui du lendemain
next_close = df["Close"].shift(-1)

valid = df["Close"].notna() & next_close.notna()

df["Direction"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="string"
)

df.loc[
    valid & (next_close > df["Close"]),
    "Direction"
] = "Hausse"

df.loc[
    valid & (next_close < df["Close"]),
    "Direction"
] = "Baisse"

df.loc[
    valid & (next_close == df["Close"]),
    "Direction"
] = "Stable"


# ==========================================
# 5. ANALYSE STATISTIQUE DE DIRECTION
# ==========================================

print("\n" + "_" * 60)
print("VARIABLE : Direction")
print("_" * 60)

direction = df["Direction"].dropna()

if len(direction) == 0:
    raise ValueError("Aucune observation valide pour Direction.")

ordre = ["Hausse", "Baisse", "Stable"]

# Effectifs
effectifs_direction = direction.value_counts().reindex(
    ordre,
    fill_value=0
)

# Frequences
frequences_direction = effectifs_direction / len(direction)

# Pourcentages
pourcentages_direction = frequences_direction * 100

# Tableau recapitulatif
tableau_direction = pd.DataFrame({
    "Effectif": effectifs_direction,
    "Fréquence": frequences_direction,
    "Pourcentage (%)": pourcentages_direction
})

print("\nEffectifs et fréquences :")
print(tableau_direction.round(4).to_string())

# Mode
modes = direction.mode().tolist()

print("\nClasse(s) modale(s) :")

for mode in modes:
    print(
        f"{mode} : "
        f"{effectifs_direction[mode]} observations"
    )

print("\nEffectif total :", len(direction))

print(
    "Valeurs manquantes :",
    df["Direction"].isna().sum()
)


# ==========================================
# 6. QUESTION 2 : NORMALISATION Z-SCORE
# ==========================================

print("\n" + "_" * 60)
print("NORMALISATION DES VARIABLES NUMERIQUES")
print("_" * 60)

variables_numeriques = ["Open", "Close"]

resultats_normalisation = {}

for variable in variables_numeriques:

    x = pd.to_numeric(df[variable], errors="coerce")

    moyenne = x.mean()
    ecart_type = x.std(ddof=0)

    if pd.notna(ecart_type) and ecart_type != 0:

        # Formule Z-score
        df[variable + "_Zscore"] = (
            x - moyenne
        ) / ecart_type

    else:
        df[variable + "_Zscore"] = np.nan

    # Verifier les resultats
    resultats_normalisation[variable] = {
        "Moyenne originale": moyenne,
        "Ecart-type original": ecart_type,
        "Moyenne normalisée": df[variable + "_Zscore"].mean(),
        "Ecart-type normalisé": df[variable + "_Zscore"].std(ddof=0)
    }

    print(f"\nVariable : {variable}")

    print("Moyenne originale :", round(moyenne, 4))
    print("Ecart-type original :", round(ecart_type, 4))

    print(
        "Moyenne après normalisation :",
        round(df[variable + "_Zscore"].mean(), 4)
    )

    print(
        "Ecart-type après normalisation :",
        round(df[variable + "_Zscore"].std(ddof=0), 4)
    )


# Tableau comparatif de normalisation
tableau_normalisation = pd.DataFrame(
    resultats_normalisation
)

print("\nTABLEAU DE NORMALISATION")
print(tableau_normalisation.round(4).to_string())


# Direction : aucune normalisation
print("\nVariable Direction :")
print("Aucune normalisation nécessaire.")
print("Variable qualitative nominale.")


# ==========================================
# 7. VISUALISATION : EFFECTIFS DIRECTION
# ==========================================

plt.figure(figsize=(8, 5))

effectifs_direction.plot(kind="bar")

plt.title("Répartition des directions du cours Apple")
plt.xlabel("Direction")
plt.ylabel("Nombre d'observations")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================
# 8. VISUALISATION : POURCENTAGES DIRECTION
# ==========================================

plt.figure(figsize=(8, 5))

pourcentages_direction.plot(kind="bar")

plt.title("Répartition des directions en pourcentage")
plt.xlabel("Direction")
plt.ylabel("Pourcentage (%)")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================
# 9. VISUALISATION : OPEN ET CLOSE
# ==========================================

for variable in ["Open", "Close"]:

    x = pd.to_numeric(df[variable], errors="coerce").dropna()

    # Histogramme
    plt.figure(figsize=(8, 5))

    plt.hist(x, bins="sturges", edgecolor="black")

    plt.title(f"Distribution de la variable {variable}")
    plt.xlabel("Prix ($)")
    plt.ylabel("Effectif")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 4))

    plt.boxplot(x, vert=False)

    plt.title(f"Boîte à moustaches de {variable}")
    plt.xlabel("Prix ($)")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==========================================
# 10. APERCU DES DONNEES
# ==========================================

print("\nAPERÇU DES DONNEES")

print(
    df[
        ["Date", "Open", "Close", "Direction",
         "Open_Zscore", "Close_Zscore"]
    ].head(10).to_string(index=False)
)


# ==========================================
# 11. EXPORTATION DES RESULTATS
# ==========================================

df.to_csv("apple_stock_direction.csv", index=False)

tableau_direction.to_csv("analyse_direction.csv")

tableau_statistiques.to_csv("statistiques_open_close.csv")

tableau_normalisation.to_csv("normalisation.csv")

print("\nFichiers exportés avec succès !")

print("- apple_stock_direction.csv")
print("- analyse_direction.csv")
print("- statistiques_open_close.csv")
print("- normalisation.csv")
