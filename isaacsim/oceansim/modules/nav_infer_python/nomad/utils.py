
import os
import sys
import io
# import matplotlib.pyplot as plt



# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import yaml
import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional

# models
# from vint_train.models.gnm.gnm import GNM
# from vint_train.models.vint.vint import ViNT

# from vint_train.models.vint.vit import ViT
from .vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from .vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from .diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
# from vint_train.data.data_utils import IMAGE_ASPECT_RATIO
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training
# LOAD DATA CONFIG
DATA_CONFIG = "./configs/infer/data.yaml"
with open(os.path.dirname(__file__) + "/configs/train/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])





def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    

    if config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


# def msg_to_pil(msg: Image) -> PILImage.Image:
#     img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
#         msg.height, msg.width, -1)
#     pil_image = PILImage.fromarray(img)
#     return pil_image


# def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
#     img = np.asarray(pil_img)  
#     ros_image = Image(encoding=encoding)
#     ros_image.height, ros_image.width, _ = img.shape
#     ros_image.data = img.ravel().tobytes() 
#     ros_image.step = ros_image.width
#     return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data