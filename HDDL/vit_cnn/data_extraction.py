import os
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import seaborn as sns


def compute_datasets_dataloaders(dataset_path,batch_size,train_size=0.7,val_size=0.1,data_augmentation=False,strong_data_augmentation=False):

    #transformations
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #valeurs de images net, à eventuelleme adapter en fonction de nos data surtout pour les vit
    ])

    full_dataset = ImageFolder(root=dataset_path, transform=transform)

    if (data_augmentation or strong_data_augmentation):

        if strong_data_augmentation :

            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                # Automated Augmentation (Replaces Rotation/ColorJitter), applies random strong distortions (shear, contrast, sharpness, etc.)
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #standard values used by ImageNet
                
                #  Random Erasing (Regularization) randomly blacks out a rectangle of the image
                transforms.RandomErasing(p=0.25) 
            ])
        else :
        #data augmentation pour le train
            transform_train = transforms.Compose([
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(224, scale=(0.08,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #valeurs de images net, à eventuelleme adapter en fonction de nos data surtout pour les vit
            ])

        augmented_full_dataset = ImageFolder(root=dataset_path, transform=transform_train)
        full_dataset = ImageFolder(root=dataset_path, transform=transform)

        np.random.seed(42)

        #division en train, validation et test, sans data augmentation sur  validation et test
        total_size = len(augmented_full_dataset)
        indices = list(range(total_size))
        split = int((train_size+val_size) * total_size)
        np.random.shuffle(indices)

        train_idx, test_idx = indices[:split], indices[split:]

        split = int(train_size * total_size)
        np.random.shuffle(train_idx)

        train_idx, val_idx = train_idx[:split], train_idx[split:]
    
        """
        Vérification que les ensembles sont bien disjoints : ok
        print(set(train_idx) & set(test_idx))
        print(set(train_idx) & set(val_idx))
        print(set(val_idx) & set(test_idx))
        """

        trainset = torch.utils.data.Subset(augmented_full_dataset, train_idx)
        valset = torch.utils.data.Subset(full_dataset, val_idx)
        testset = torch.utils.data.Subset(full_dataset, test_idx)

    else :

        total_len = len(full_dataset)
        train_len = int(0.7 * total_len)
        val_len   = int(0.1 * total_len)
        test_len  = total_len - train_len - val_len  


        generator = torch.Generator().manual_seed(42)

        trainset, valset, testset = random_split(full_dataset, [train_len, val_len, test_len],
            generator=generator
        )


    print(f"Total: {len(full_dataset)}")
    print(f"Train: {len(trainset)} images")
    print(f"Val:   {len(valset)} images")
    print(f"Test:  {len(testset)} images")


    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True,  num_workers=4)
    valoader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return full_dataset,trainset,valset,testset,trainloader,valoader,testloader

def compute_datasets_dataloaders_imagenet(dataset_path, batch_size, data_augmentation=False, strong_data_augmentation=False):
    """
    Charge les données Tiny ImageNet (Train et Val uniquement).
    Le dossier 'val' doit avoir été structuré par classe au préalable.
    """
    
    # 1. Définition des transformations
    
    # Normalisation standard ImageNet
    mean_stats = [0.485, 0.456, 0.406]
    std_stats  = [0.229, 0.224, 0.225]

    # Transformation de base (Validation) : Juste Redimensionnement + Normalisation
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)), # Obligatoire pour ViT / ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_stats, std=std_stats)
    ])

    # Transformation pour le Train (Selon le niveau d'augmentation choisi)
    if strong_data_augmentation:
        print(f"Mode: Strong Data Augmentation (RandAugment + Erasing)")
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            # RandAugment est très efficace pour les ViT
            transforms.RandAugment(num_ops=2, magnitude=9), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_stats, std=std_stats),
            transforms.RandomErasing(p=0.25) 
        ])
        
    elif data_augmentation:
        print(f"Mode: Standard Data Augmentation")
        transform_train = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_stats, std=std_stats)
        ])
        
    else:
        print(f"Mode: No Data Augmentation (Baseline)")
        transform_train = transform_val

    # 2. Chargement des Datasets via ImageFolder
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')

    # ImageFolder scanne les sous-dossiers pour créer les classes
    trainset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    valset = datasets.ImageFolder(root=val_dir, transform=transform_val)

    # 3. Création des DataLoaders
    # num_workers=4 et pin_memory=True accélèrent le transfert vers le GPU
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valoader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("-" * 30)
    print(f"Données chargées depuis : {dataset_path}")
    print(f"Classes détectées : {len(trainset.classes)}")
    print(f"Images d'entraînement : {len(trainset)}")
    print(f"Images de validation : {len(valset)}")
    print("-" * 30)

    return trainset, valset, trainloader, valoader


def plot_dataset_distribution(full_dataset, train_set, val_set, test_set):

    
    class_names = full_dataset.classes
    all_targets = np.array(full_dataset.targets)
    
    def get_counts(subset, split_name):
        subset_targets = all_targets[subset.indices]
        unique, counts = np.unique(subset_targets, return_counts=True)
        
        count_dict = {class_names[i]: 0 for i in range(len(class_names))}
        for idx, count in zip(unique, counts):
            count_dict[class_names[idx]] = count
            
        data = []
        for cls, count in count_dict.items():
            data.append({'Classe': cls, 'Nombre': count, 'Set': split_name})
        return data


    data_list = []
    data_list.extend(get_counts(train_set, 'Train'))
    data_list.extend(get_counts(val_set, 'Val'))
    data_list.extend(get_counts(test_set, 'Test'))
    
    df = pd.DataFrame(data_list)
    
    fig, axes = plt.subplots(2, 1, figsize=(50, 25))
    
    df_total = df.groupby('Classe')['Nombre'].sum().reset_index()
    sns.barplot(data=df_total, x='Classe', y='Nombre', ax=axes[0], palette='viridis')
    axes[0].set_title("Répartition dataset complet", fontsize=24, pad=20)
    axes[0].set_ylabel("Nombre d'images",fontsize=20)
    axes[0].tick_params(axis='x', rotation=45,labelsize=20)
    axes[0].tick_params(axis='y', labelsize=20)
    
    for container in axes[0].containers:
        axes[0].bar_label(container)

    sns.barplot(data=df, x='Classe', y='Nombre', hue='Set', ax=axes[1], palette='magma')
    axes[1].set_title("Répartition des classes dans les trainset / valset / testset", fontsize=24, pad=20)
    axes[1].set_ylabel("Nombre d'images", fontsize=20)
    axes[1].tick_params(axis='x', rotation=45,labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)
    

    axes[1].legend(fontsize=20, title="Jeux de données", title_fontsize=20)
    plt.tight_layout()
    plt.show()


    print("--- Récapitulatif ---")
    print(df.groupby('Set')['Nombre'].sum())