# KU-DEEP-LEARNING Term Project

### This code is on-going

## Install
```
pip install -r requrements.txt
```

## Files

### 0.teleop.ipynb
Contains keyboard teleoperation demo.

Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of rthe otations.

SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.


### 1.Visualize.ipynb

It contains downloading dataset from huggingface and visualizing it.
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
root = './dataset/demo_data'
dataset = LeRobotDataset('Jeongeun/deep_learning_2025',root = root )
```
Running this code will download the dataset independatly.

### 2.transform.ipynb
Define the action and observation space for the environment. 
```python
action_type = 'delta_joint'  # Options: 'joint','delta_joint, 'delta_eef_pose', 'eef_pose'
proprio_type = 'eef_pose' # Options: 'joint', 'eef_pose'
observation_type = 'image' # options: 'image', 'object_pose'
image_aug_num = 2  # Number of augmented images to generate per original image
transformed_dataset_path = './dataset/transformed_data'
```

Based on this configuration, it will transform the actions into the action_type and create new dataset for training. 

- action_type: representation of the actions. Options: 'joint','delta_joint','eef_pose','delta_eef_pose'
- proprio_type: representations of propriocotative informations. Options: eef_pose, joint_pos
- observation_type: whether to use image of a object position informations. Options: 'image','objet_pose'
- image_aug_num: the number of augmented trajectories to make when you are using image features

You can just use the python script to do this as well. 

```
python transform.py --action_type delta_eef_pose --proprio_type eef_pose --observation_type image --image_aug_num 2
```

## Others

### Data collection with leader arm
First, launch the ros2 package from ROBOTIS to turn on the leader. This requires ROS2. 
```
ros2 launch open_manipulator_bringup hardware_y_leader.launch.py
```
Then, with the other terminal run 
```
python leader.py
```

Finally, on the third terminal, run
```
python collect_data.py
```
to collect the data!