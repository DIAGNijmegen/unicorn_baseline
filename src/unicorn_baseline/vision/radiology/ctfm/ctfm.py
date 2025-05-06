from lighter_zoo import SegResNet
from monai.transforms import Compose, LoadImage, EnsureType, ScaleIntensityRange, Orientation, Resize
from monai.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch


def load_model(model_path):
    # Load pre-trained model
    return SegResNet.from_pretrained(
        model_path
    )


def load_data(data): 
    train_transforms = Compose([ 
        EnsureType(),                        
        Orientation(axcodes="SPL"),       
        ScaleIntensityRange(
            a_min=-1024,    
            a_max=2048,    
            b_min=0,        
            b_max=1,        
            clip=True       
        ),
     ])

    train_ds = Dataset(data=data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    return train_loader

def encode(model, patches): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    #expand patch to match encoder input requirements
    patch_array = np.expand_dims(patches, axis=(0, 1))
    
    train_loader = load_data(patch_array)

    model.eval() 
    with torch.no_grad():
        for input in train_loader:
            input = input.to(device)
            output = model.encoder(input) 
            #average pool and flatten the output to fit feature vector requirements
            output_flat = adaptive_pool(output[-1])
            out_flat = output_flat.flatten(start_dim =0)

    return out_flat.cpu().detach().numpy().tolist()
