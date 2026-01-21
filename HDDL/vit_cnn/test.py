from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_confusion_matrix(true_lbl, pred_lbl, class_names)
    cm = confusion_matrix(true_lbl, pred_lbl)

    # Créer un DataFrame complet avec les noms de races
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Remplacer les bonnes prédictions par NaN pour n'afficher que les erreurs
    cm_errors = cm_df.mask(np.eye(len(cm_df), dtype=bool))

    #    Affichage
    plt.figure(figsize=(100,100))
    sns.heatmap(cm_errors, annot=True, fmt=".0f", cmap="Reds", cbar=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité terrain")
    plt.title("Erreurs de prédiction (compact avec NaN pour diagonale)")
    plt.show()

