import torch.nn as nn
import torch.nn.functional as F
import torch

class LanguageConditionedPolicy(nn.Module):
    def __init__(self, language_dim, action_dim):
        super(LanguageConditionedPolicy, self).__init__()
        self.language_fc1 = nn.Linear(language_dim, 256)
        self.image_fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_dim)

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128*23*23, 512)
        )
        
    def forward(self, language_emb, image_features):
        x = F.relu(self.language_fc1(language_emb))
        # y = F.relu(self.image_fc1(self.visual_encoder(image)))
        y = F.relu(self.image_fc1(image_features))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        action = self.fc5(z)
        return action