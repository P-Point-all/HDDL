import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from data_extraction import compute_datasets_dataloaders
from models.simpleCNN import CNN_classifier
from train import train_and_validate, plot_training

if __name__=='__main__':

    dataset_path="./food-101/images/"
    batch_size=64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    food_fulldataset,food_trainset,food_valset,food_testset,food_trainloader,food_valoader,food_testloader=compute_datasets_dataloaders(dataset_path,batch_size,train_size=0.7,val_size=0.1,data_augmentation=True)
    class_names = food_fulldataset.classes

    nb_classes=len(class_names)

    # Initialisation du modèle
    # num_classes=101 pour Food-101
    model = CNN_classifier(num_classes=nb_classes, im_size=224).to(device)

    checkpoint_path="weights/food101_SimpleCNN"

    config={'lr': 0.001, 'weight_decay': 1e-05}
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    nb_epochs=20
    
    
    best_acc,history_train_loss,history_val_acc=train_and_validate(model, config, food_trainloader, food_valoader, 101, device, epochs=20, im_size=224, checkpoint_path=checkpoint_path)

    plot_training(nb_epochs,history_train_loss,history_val_acc)