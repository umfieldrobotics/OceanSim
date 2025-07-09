# import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# import matplotlib.pyplot as plt
import yaml

# ROS
# import rospy
# from sensor_msgs.msg import Image
# from std_msgs.msg import Bool, Float32MultiArray
from .utils import to_numpy, transform_images, load_model, get_action

# from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
# import argparse
import yaml
import time


# UTILS
# from topic_names import (IMAGE_TOPIC,
#                         WAYPOINT_TOPIC,
#                         SAMPLED_ACTIONS_TOPIC)


# CONSTANTS
# MODEL_WEIGHTS_PATH = "../model_weights"
# ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = os.path.dirname(__file__) + "/configs/infer/models.yaml"
# with open(ROBOT_CONFIG_PATH, "r") as f:
#     robot_config = yaml.safe_load(f)
# MAX_V = robot_config["max_v"]
# MAX_W = robot_config["max_w"]
# RATE = robot_config["frame_rate"] 

class NoMadModel:
    def __init__(self, model:str = "nomad", waypoint:int = 2, num_samples:int = 8):

        self.model = model
        self.waypoint = waypoint
        self.num_samples = num_samples
        # Load the model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[self.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)

        self.context_size = model_params["context_size"]

        # load model weights
        ckpth_path = model_paths[self.model]["ckpt_path"]
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        self.model = load_model(
            ckpth_path,
            model_params,
            self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_diffusion_iters = model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.image_size = model_params["image_size"]
        self.len_traj_pred = model_params["len_traj_pred"]
        self.normalize = model_params["normalize"]
        self.context_queue = []
    
    def callback_obs(self, img : np.ndarray):
        obs_img = PILImage.fromarray(img)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)
    
    
    
    def infer(self):
        if (len(self.context_queue) > self.context_size):

            obs_images = transform_images(self.context_queue, self.image_size, center_crop=False)
            obs_images = obs_images.to(self.device)
            fake_goal = torch.randn((1, 3, *self.image_size)).to(self.device)
            mask = torch.ones(1).long().to(self.device) # ignore the goal

            # infer action
            with torch.no_grad():
                # encoder vision features
                obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
                
                # (B, obs_horizon * obs_dim)
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(self.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(self.num_samples, 1, 1)
                
                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (self.num_samples, self.len_traj_pred, 2), device=self.device)
                naction = noisy_action

                # init scheduler
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                start_time = time.time()
                for k in self.noise_scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = self.model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                print("time elapsed:", time.time() - start_time)

            naction = to_numpy(get_action(naction))
            
            # sampled_actions_msg = Float32MultiArray()
            sampled_actions = np.concatenate((np.array([0]), naction.flatten()))
            # sampled_actions_pub.publish(sampled_actions_msg)

            naction = naction[0] # change this based on heuristic

            chosen_waypoint = naction[self.waypoint]

            # if model_params["normalize"]:
            #     chosen_waypoint *= (MAX_V / RATE)
            waypoint = chosen_waypoint

            return sampled_actions, waypoint









