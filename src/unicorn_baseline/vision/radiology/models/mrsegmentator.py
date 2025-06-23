import torch.nn as nn
import json
from dynamic_network_architectures.architectures.unet import PlainConvUNet
import torch
from monai.transforms import (
    Compose,
    EnsureType,
)
from monai.data import Dataset, DataLoader
import numpy as np


def load_model_mr(model_dir):
    with open(f"{model_dir}/plans.json") as f:
        plans = json.load(f)

    input_channels = 1  # len(dataset_json['channel_names']
    n_stages = len(
        plans["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"]
    )  # len(self.configuration["n_conv_per_stage_encoder"])
    features_per_stage = [
        min(
            plans["configurations"]["3d_fullres"]["UNet_base_num_features"] * 2**i,
            plans["configurations"]["3d_fullres"]["unet_max_num_features"],
        )
        for i in range(
            len(plans["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"])
        )
    ]  # self.UNet_max_features_3d = 320, n_stages
    conv_op = (
        nn.Conv3d
    )  # len(plans['configurations']['3d_fullres']['patch_size']) == 3 -> nn.Conv3d
    kernel_sizes = plans["configurations"]["3d_fullres"]["conv_kernel_sizes"]
    strides = plans["configurations"]["3d_fullres"]["pool_op_kernel_sizes"]
    n_conv_per_stage = plans["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"]
    num_classes = 41
    n_conv_per_stage_decoder = plans["configurations"]["3d_fullres"][
        "n_conv_per_stage_decoder"
    ]

    model = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={"inplace": True},
    )
    weights = torch.load(
        f"{model_dir}/fold_0/checkpoint_final.pth",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict(weights["network_weights"])
    return model


def load_data(data):
    train_transforms = Compose(
        [
            EnsureType(dtype=torch.float32),
        ]
    )

    train_ds = Dataset(data=data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    return train_loader


def encode_mr(model, patches):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    # expand patch to match encoder input requirements
    patch_array = np.expand_dims(patches, axis=(0, 1))

    train_loader = load_data(patch_array)

    model.eval()
    with torch.no_grad():
        for input in train_loader:
            input = input.to(device)
            output = model.encoder(input)
            # average pool and flatten the output to fit feature vector requirements
            output_flat = adaptive_pool(output[-1])
            out_flat = output_flat.flatten(start_dim=0)

    return out_flat.cpu().detach().numpy().tolist()
