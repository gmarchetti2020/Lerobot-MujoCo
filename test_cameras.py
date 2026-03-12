import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append('.')
from src.env.env_ur5 import RILAB_OMY_ENV
import json

with open('configs/train_ur5.json', 'r') as f:
    cfg = json.load(f)

env = RILAB_OMY_ENV(cfg, vis_mode='teleop')
agent_img, wrist_img = env.grab_image(return_side=False)

Image.fromarray(agent_img).save('agent_view.png')
Image.fromarray(wrist_img).save('wrist_view.png')
print("Saved agent_view.png and wrist_view.png")
