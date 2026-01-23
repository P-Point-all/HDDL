
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_extraction import compute_datasets_dataloaders
from models.simpleCNN import CNN_classifier
from models.ResNetCNN import ComplexCNN
from models.ViT import ViT

from torch.cuda.amp import autocast, GradScaler
import itertools
import json

import torch.optim.lr_scheduler as lr_scheduler

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os 

def train_and_validate(model, optimizer, train_loader, val_loader, device, checkpoint_path,save_info_path, epochs=5, save_model=True,scheduler=None):
    

    """
    Entraine le modèle et sauvegarde les poids à chaque fois que l'accuracy sur le jeu de validation est maximale. 
    A la fin de l'entrainement enregistre un dictionnaire contenant toutes les infos pertientes : historique de la loss de train et validation, 
    historique de l'accuracy de train et validation, temps total, temps moyen d'une époque, temps de chaque époque, nomtre total d'époques, poids du modèle, de l'optimiseur, l'accuracy et 
        'history_val_loss': [],
        'history_train_acc': [],
        'history_val_acc': [],
        'total_time_seconds': 0,
        'avg_epoch_time_seconds': 0,
        'epochs_times': [], 
        'total_epochs': 0,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'best_acc': None,
        'scaler_state_dict':None
    """

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))


    if os.path.exists(save_info_path):
        training_info = torch.load(save_info_path, map_location='cpu')
    
        # Charger l'état du modèle
        model.load_state_dict(training_info['model_state_dict'])
        print("Poids du modèle chargés avec succès.")
    
        # Charger l'état de l'optimiseur
        optimizer.load_state_dict(training_info['optimizer_state_dict'])
        print("État de l'optimiseur chargé avec succès.")

        scaler.load_state_dict(training_info['scaler_state_dict'])

        if scheduler and 'scheduler_state_dict' in training_info:
            scheduler.load_state_dict(training_info['scheduler_state_dict'])
            print("État du scheduler chargé avec succès.")

        best_acc = training_info.get('best_acc')
        start_epoch=training_info['total_epochs']

        # Reprendre l'époque de départ
        print(f"Reprise de l'entraînement à partir de l'époque {start_epoch}. (Best Acc: {best_acc:.2%})") 

    else:
        print(f"Début de l'entraînement à partir de zéro.")
        best_acc = 0.0
        start_epoch =0
        
        training_info = {
        'history_train_loss': [],
        'history_val_loss': [],
        'history_train_acc': [],
        'history_val_acc': [],
        'total_time_seconds': 0,
        'avg_epoch_time_seconds': 0,
        'epochs_times': [], 
        'total_epochs': 0,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'best_acc': 0,
        'scaler_state_dict':None
        }
       
    
    
    # Top départ global
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time() # Top départ époque
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        correct_train, total_train = 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Ép. {start_epoch+epoch+1}/{start_epoch+epochs} [Train]", mininterval=2.0)
        
        for images, labels in train_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            total_train += labels.size(0)
            _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = correct_train / total_train
        
        training_info['history_train_loss'].append(epoch_loss)
        training_info['history_train_acc'].append(epoch_acc)
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        correct_val, total_val = 0, 0
        
        with torch.no_grad():
            with autocast(enabled=(device.type == 'cuda')):
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss_v = criterion(outputs, labels)
                    val_loss += loss_v.item()
                    
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        training_info['history_val_loss'].append(epoch_val_loss)
        training_info['history_val_acc'].append(val_acc)
        
        # --- TEMPS ---
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        training_info['epochs_times'].append(epoch_duration)
        training_info['total_epochs']+=1
        
        print(f"--> Fin Ép. {start_epoch+epoch+1} ({epoch_duration:.1f}s) : Train Loss = {epoch_loss:.4f} Acc = {epoch_acc:.2%} | Val Loss = {epoch_val_loss:.4f} Acc = {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            if save_model:
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': val_acc,
                    'scaler_state_dict': scaler.state_dict()
                }
                # On ajoute le scheduler s'il existe
                if scheduler:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                print('Meilleure val accuracy, modèle sauvegardé !')

        #si on a un scheduler, maj du scheduler

        if (scheduler):
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"LR : {current_lr:.6f}")



    # --- FIN DE L'ENTRAINEMENT ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
   
    training_info['total_time_seconds']+=total_duration
    training_info['avg_epoch_time_seconds']=sum(training_info['epochs_times']) / len(training_info['epochs_times']) 
    


    # Sauvegarde du dictionnaire d'infos
    if save_model:
        folder = os.path.dirname(save_info_path)
        if folder: 
            os.makedirs(folder, exist_ok=True)
        
        training_info['model_state_dict']= model.state_dict()
        training_info['optimizer_state_dict']= optimizer.state_dict()
        training_info['best_acc']= best_acc
        training_info['scaler_state_dict']= scaler.state_dict()
        training_info['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(training_info, save_info_path)

        print(f"\n[INFO] Résumé de l'entraînement sauvegardé dans : {save_info_path}")
        print(f"[INFO] Temps total : {total_duration/60:.1f} min | Moyen/Ép : {training_info['avg_epoch_time_seconds']:.1f} sec")
            
    return best_acc, training_info

def train_and_validate_corrected(model, optimizer, train_loader, val_loader, device, checkpoint_path, save_info_path, epochs=5, save_model=True, scheduler=None):
    
    # 1. Mise à jour de la syntaxe AMP et du Scaler
    criterion = nn.CrossEntropyLoss()
    # Nouvelle syntaxe recommandée par PyTorch
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # --- CHARGEMENT DU CHECKPOINT (Reprise) ---
    # Note : On vérifie si le fichier d'infos existe pour reprendre l'historique
    if os.path.exists(save_info_path):
        print(f"--> Chargement des infos d'entraînement depuis : {save_info_path}")
        training_info = torch.load(save_info_path, map_location=device) # Charge sur le bon device
    
        model.load_state_dict(training_info['model_state_dict'])
        optimizer.load_state_dict(training_info['optimizer_state_dict'])
        scaler.load_state_dict(training_info['scaler_state_dict'])

        if scheduler and 'scheduler_state_dict' in training_info:
            scheduler.load_state_dict(training_info['scheduler_state_dict'])

        best_acc = training_info.get('best_acc', 0.0)
        start_epoch = training_info.get('total_epochs', 0)

        print(f"--> Reprise à l'époque {start_epoch}. (Best Acc précédente: {best_acc:.2%})") 

    else:
        # Si pas de fichier info, on regarde si on a juste un checkpoint de poids
        if os.path.exists(checkpoint_path):
            print(f"--> Attention : 'save_info_path' introuvable, mais 'checkpoint_path' existe.")
            print(f"--> Chargement des poids uniquement depuis {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            best_acc = ckpt.get('acc', 0.0)
            # On ne peut pas récupérer l'historique complet, donc on repart à 0 pour les stats
            # mais avec un modèle pré-entrainé.
        else:
            print(f"--> Début de l'entraînement à partir de zéro.")
            best_acc = 0.0
        
        start_epoch = 0
        
        training_info = {
            'history_train_loss': [], 'history_val_loss': [],
            'history_train_acc': [], 'history_val_acc': [],
            'total_time_seconds': 0, 'avg_epoch_time_seconds': 0,
            'epochs_times': [], 'total_epochs': 0,
            'model_state_dict': None, 'optimizer_state_dict': None,
            'best_acc': 0, 'scaler_state_dict':None
        }

    total_start_time = time.time()
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        current_epoch_display = start_epoch + epoch + 1
        epoch_start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        correct_train, total_train = 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Ép. {current_epoch_display} [Train]", mininterval=2.0)
        
        for images, labels in train_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Nouvelle syntaxe AMP
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            # --- CRUCIAL : GRADIENT CLIPPING (Pour éviter NaN) ---
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # -----------------------------------------------------

            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            total_train += labels.size(0)
            _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = correct_train / total_train
        
        training_info['history_train_loss'].append(epoch_loss)
        training_info['history_train_acc'].append(epoch_acc)
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        correct_val, total_val = 0, 0
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss_v = criterion(outputs, labels)
                    val_loss += loss_v.item()
                    
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        training_info['history_val_loss'].append(epoch_val_loss)
        training_info['history_val_acc'].append(val_acc)
        
        # --- TEMPS ---
        epoch_duration = time.time() - epoch_start_time
        training_info['epochs_times'].append(epoch_duration)
        training_info['total_epochs'] += 1 # On incrémente le compteur total
        
        print(f"--> Fin Ép. {current_epoch_display} ({epoch_duration:.1f}s) : Train Loss = {epoch_loss:.4f} Acc = {epoch_acc:.2%} | Val Loss = {epoch_val_loss:.4f} Acc = {val_acc:.2%}")

        # --- SAUVEGARDE BEST MODEL ---
        if val_acc > best_acc:
            best_acc = val_acc
            if save_model:
                # CREATION DOSSIER AUTOMATIQUE
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': val_acc,
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': current_epoch_display # On sauvegarde l'époque
                }
                if scheduler:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                
                torch.save(checkpoint_data, checkpoint_path)
                print('★ Meilleure val accuracy ! Modèle sauvegardé.')

        # --- SCHEDULER ---
        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
                # Hack pour afficher le LR avec ReduceLROnPlateau
                current_lr = optimizer.param_groups[0]['lr']
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            
            print(f"LR : {current_lr:.6f}")

    # --- FIN ---
    total_duration = time.time() - total_start_time
    training_info['total_time_seconds'] += total_duration
    if training_info['epochs_times']:
        training_info['avg_epoch_time_seconds'] = sum(training_info['epochs_times']) / len(training_info['epochs_times'])

    # Sauvegarde finale
    if save_model:
        # CREATION DOSSIER AUTOMATIQUE
        os.makedirs(os.path.dirname(save_info_path), exist_ok=True)
        
        training_info['model_state_dict'] = model.state_dict()
        training_info['optimizer_state_dict'] = optimizer.state_dict()
        training_info['best_acc'] = best_acc
        training_info['scaler_state_dict'] = scaler.state_dict()
        if scheduler:
            training_info['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(training_info, save_info_path)
        print(f"\n[INFO] Résumé complet sauvegardé dans : {save_info_path}")

    return best_acc, training_info




def optim_param(model_class,nb_classes,param_grid,optimizer_algo,trainloader,valoader, device,saving_path):

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    best_overall_acc = 0
    best_config = None

    print(f"Lancement de la GridSearch : {len(combinations)} combinaisons à tester.")

    for i, config in enumerate(combinations):
        print(f"\n--- Test {i+1}/{len(combinations)} | Params: {config} ---")

        model=model_class(num_classes=nb_classes).to(device)

        optimizer = optimizer_algo(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )

        # On appelle la fonction pour le CNN Simple
        acc, _ = train_and_validate(model, optimizer, trainloader, valoader, device,"","" ,epochs=config["epochs"],save_model=False)
        
        config['val_accuracy'] = acc
        results.append(config)
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_config = config
        
        print(f"Résultat : Accuracy Val = {acc*100:.2f}%")

        #del model 
        torch.cuda.empty_cache()    

    # 3. Sauvegarde des résultats pour ton rapport
    with open(saving_path, 'w') as f:
        json.dump(results, f, indent=4)

    return best_config





def plot_training(training_info):

    nb_epochs=len(training_info["history_train_loss"])

    plt.figure(figsize=(12, 5))



    # Courbe de Loss (Train)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, nb_epochs+1), training_info["history_train_loss"], label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, nb_epochs+1), training_info["history_val_loss"], label='Val Loss', color='green', marker='o')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Courbe d'Accuracy (Validation)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, nb_epochs+1), training_info["history_train_acc"], label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, nb_epochs+1), training_info["history_val_acc"], label='Val Accuracy', color='green', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



    