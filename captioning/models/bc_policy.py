import torch.nn as nn
import torch.nn.functional as F
import torch

class LanguageConditionedPolicy(nn.Module):
    def __init__(self, language_dim, action_dim):
        super(LanguageConditionedPolicy, self).__init__()

        self.language_fc1 = nn.Linear(language_dim, 256)
        self.image_static_fc1 = nn.Linear(512, 256)
        self.image_gripper_fc1 = nn.Linear(512, 256)
        self.obs_fc1 = nn.Linear(15, 64)

        self.fc2 = nn.Linear(832, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, action_dim)

        
    def forward(self, language_emb, image_features_static, image_features_gripper, robot_obs):
        lang = F.relu(self.language_fc1(language_emb))
        obs = F.relu(self.obs_fc1(robot_obs))

        static = F.relu(self.image_static_fc1(image_features_static))
        gripper = F.relu(self.image_gripper_fc1(image_features_gripper))
        img = torch.cat((static, gripper), dim=1)

        z = torch.cat((lang, img, obs), dim=1)
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        action = self.fc6(z)
        return action