import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Première convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False) # le biais serait redondant avec celui utilisé dans le BatchNorm
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        # Deuxième convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Le "Shortcut" (Le pont)
        # Si on change la taille de l'image (stride > 1) ou le nombre de canaux, 
        # on doit adapter l'entrée pour pouvoir l'additionner à la sortie.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # L'addition magique : on ajoute l'entrée originale au résultat
        out += self.shortcut(identity)
        out = self.gelu(out)
        
        return out

# Le bloc résiduel est la brique élémentaire de notre ResNet : son rôle est d'apprendre une correction (un résidu) par rapport à l'entrée plutôt
# que quelque chose de complètement nouveau.

class ComplexCNN(nn.Module):
    def __init__(self, num_classes, im_size=224):
        super(ComplexCNN, self).__init__()
        
        self.in_channels = 64
        
        # Entrée initiale : 224x224 -> 56x56 par le biais de la couche de départ. Elle permet d'avoir des blocs résiduels sur des objets plus petits.
        self.start_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # Filtres 7x7 pour avoir un recul global sur l'image
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Couches résiduelles (de plus en plus de canaux, image de plus en plus petite)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        
        # Sortie globale
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        # Le premier bloc de la couche peut changer la résolution (stride)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # Les blocs suivants gardent la même résolution
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.start_layer(x) # [Batch, 64, 56, 56]
        
        x = self.layer1(x)      # [Batch, 64, 56, 56]
        x = self.layer2(x)      # [Batch, 128, 28, 28]
        x = self.layer3(x)      # [Batch, 256, 14, 14]
        x = self.layer4(x)      # [Batch, 512, 7, 7]
        
        x = self.gap(x)         # [Batch, 512, 1, 1]
        x = torch.flatten(x, 1) # [Batch, 512]
        x = self.classifier(x)  # [Batch, num_classes]
        
        return x

# Ce CNN est plus complexe : il est constitué de 4 blocs résiduels (contenant chacun deux opérateurs convolutifs) + un premier bloc de compression.
# Il reprend un mécanisme également présent dans le ViT, celui de correction de l'image de base (avec l'addition en sortie du bloc résiduel), il
# est donc une sorte d' "entre-deux".