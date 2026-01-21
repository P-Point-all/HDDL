import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        # On utilise une conv pour découper et projeter en une seule étape
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        x = self.projection(x) # [Batch, embed_dim, 14, 14]
        x = x.flatten(2)       # [Batch, embed_dim, 196]
        x = x.transpose(1, 2)  # [Batch, 196, embed_dim] (Format Séquence)
        return x
    
# Chaque patch est projeté dans un espace d'embedding à l'aide d'une Convolution 2D

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # 1. La normalisation (on la définit ici pour l'utiliser avant l'attention)
        self.layernorm = nn.LayerNorm(embed_dim)
        
        # 2. Le mécanisme d'attention
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,      # Aide à éviter le sur-apprentissage
            batch_first=True
        )
        
        # 3. Le Dropout pour la régularisation après l'attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # --- Étape 1 : Copie pour la connexion résiduelle ---
        residual = x
        
        # --- Étape 2 : Pre-Norm ---
        # On normalise d'abord (très important pour la stabilité)
        out = self.layernorm(x)
        
        # --- Étape 3 : Self-Attention ---
        # Query = Key = Value = out
        out, _ = self.mha(out, out, out)
        
        # --- Étape 4 : Dropout et Addition Résiduelle ---
        out = self.dropout(out)
        return out + residual

# On normalise l'embedding, on les divise en plusieurs morceaux (1 pour chaque tête) et on obtient par
# calcul matriciel les Q, K, V pour chaque tête. On applique ensuite un mécanisme d'attention pour chaque
# tête, on concatène les résultats, et on multiplie finalement par une matrice. On peut éventuellement faire
# un drop-out aléatoire sur la matrice obtenue, que ce soit après le softmax ou/et à la fin pour éviter de 
# faire du sur-apprentissage sur des détails. Dans le cadre du jeu de données GTSRB, les images du jeu 
# d'entraînement correspondent à des captations successives sur vidéo donc pour une même classe, les images
# se ressemblent : le drop-out est donc utile.

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(), # Activation standard des Transformers
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Même logique que le MHA : Norm + Add
        return self.mlp(self.layernorm(x)) + x
    
# Après chaque bloc d'attention, on fait une couche de feed-forward avec normalisation, drop-out éventuel
# (extinction aléatoire des neurones et dans le résultat final). On utilise la fonction d'activation GeLU (Gaussian Error Linear Unit) qui est une
# version "lissée" et dérivable de ReLU pour préserver le gradient pour les valeurs négatives.

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=101, 
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout_mha = 0.25, dropout_mlp = 0.25):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        # 1. CLS Token et Position Embedding (Paramètres apprenables)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)*0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)*0.02)
        
        # 2. Les blocs Transformer (Séquence de blocs)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttentionBlock(embed_dim, num_heads, dropout_mha),
                MLPBlock(embed_dim, mlp_dim, dropout_mlp)                  
            ]))
        
        # 3. Tête de classification finale
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        
        # Ajout du CLS token à chaque échantillon du batch
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Ajout de l'encodage de position
        x = x + self.pos_embed
        
        for attn_block, mlp_block in self.layers:
            x = attn_block(x)
            x = mlp_block(x)
            
        # On ne garde que la sortie du CLS token pour prédire la classe
        return self.mlp_head(x[:, 0])
    
# Les paramètres d'entrée par défaut sont choisis conformément aux standards de référence proposés par Google dans le cadre de sa configuration
# "ViT-Base". Le CLS Token et le Position Embedding sont appris au même titre que les poids des réseaux. Ils ne sont pas initialisés à 0 pour
# que le modèle démarre rapidement. 
# Le CLS Token ne correspondant à rien de précis, à la suite de l'application du mécanisme d'attention, le premier vecteur va correspondre à une 
# sorte de résumé global des informations fournies par tous les autres patches. C'est donc lui que l'on choisit en entrée du réseau pour la
# prédiction de classe.