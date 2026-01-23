import torch
import torch.nn as nn

class CNN_classifier(nn.Module):
    def __init__(self, num_classes, im_size=224):
        super(CNN_classifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        # On utilise AdaptiveAvgPool2d pour forcer la sortie à 1x1 pixel par canal, permet davantage de robustesse
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes),
        )

        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        x = self.fc_layers(x)
        return x


if __name__=='__main__':
    im_size=224
    x = torch.rand(1, 3, im_size, im_size)
    net = CNN_classifier(num_classes=10, im_size=im_size)
    
    out = net(x)

    print(f"Sortie : {out.shape}")    
    
# Il s'agit d'un code de CNN très simple avec seulement 4 couches convolutionnelles
