from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_confusion_matrix(true_lbl, pred_lbl, class_names):
    cm = confusion_matrix(true_lbl, pred_lbl)

    # Créer un DataFrame complet avec les noms de races
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Remplacer les bonnes prédictions par NaN pour n'afficher que les erreurs
    cm_errors = cm_df.mask(np.eye(len(cm_df), dtype=bool))

    diff_sum_per_class = cm_errors.sum(axis=1)
    #On trie par ordre croissant
    diff_sum_per_class = diff_sum_per_class.sort_values(ascending=False)
    print("Somme des erreurs par classe :")
    for class_name, error_sum in zip(class_names, diff_sum_per_class):
        print(f"{class_name}: {error_sum}")
    #    Affichage
    plt.figure(figsize=(100,100))
    sns.heatmap(cm_errors, annot=True, fmt=".0f", cmap="Reds", cbar=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité terrain")
    plt.title("Erreurs de prédiction (compact avec NaN pour diagonale)")
    plt.show()  

