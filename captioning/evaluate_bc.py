
import torch
import numpy as np
from torchvision import transforms

from models.visual_autoencoder import VisualAutoencoder
from models.bc_policy import LanguageConditionedPolicy

from torchvision.transforms import InterpolationMode
import config as CFG
import clip
import pdb
import cv2 
import matplotlib.pyplot as plt
import hydra
from utils.utils import AttrDict


class EvaluateBCAgent():
    def __init__(self, cfg):
        # self.env = PlayTableSimEnv()
        self.env = self.get_env(cfg)
        self.device = CFG.device

        clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=True)
        self.clip_text_encoder = clip_model.encode_text
        self.clip_image_encoder = clip_model.encode_image
        #self.transform = clip_preprocess()

        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((CFG.img_size)),
            transforms.ToTensor() 
        ])

      #  self.model = BehaviourEncoder(temperature=0.07).to(CFG.device)
      #  self.model.load_state_dict(torch.load("/home/krishan/work/skillGPT/skillGPT/checkpoints/lang_table_sim_full_clip_encoding_fixed_temp_softmax_gt_2023.03.16_16:04:39_alignment_best_MARCH_BEST.pth"))
      #  self.model.eval()
      #  self.model.to(self.device)

        self.policy = LanguageConditionedPolicy(language_dim=512, action_dim=7)
        # self.policy.load_state_dict(torch.load("/media/tim/E/hulc_captioning/captioning/checkpoints/bc_policy/bc_policy_static_2/bc_policy_static-036.pt"))
        self.policy.load_state_dict(torch.load("/media/tim/E/hulc_captioning/captioning/checkpoints/bc_policy/bc_policy_gripper/bc_policy_static-009.pt"))
        self.policy = self.policy.to(self.device)

        self.action_stats = AttrDict(mean=np.array([ 0.05829782, -0.11443839,  0.5123094,   0.97989569, -0.03639337,  1.55281177, 0.01424668], dtype=np.float32),
                                     std=np.array([0.1525908,  0.09533093, 0.05158806, 2.920689, 0.10144497, 0.56220544, 0.99989851]), dtype=np.float32)

        
        # initialize the visual autoencoder
        self.visual_autoencoder = VisualAutoencoder(512)
        self.visual_autoencoder = self.visual_autoencoder.to(self.device)
        
        # load the weights
        # self.visual_autoencoder.load_state_dict(torch.load("/media/tim/E/hulc_captioning/captioning/checkpoints/image_vae/image_vae_static_full_long/image_vae_static_full_high_res-045.pt"))
        self.visual_autoencoder.load_state_dict(torch.load("/media/tim/E/hulc_captioning/captioning/checkpoints/image_vae/image_vae_gripper/image_vae_gripper_high_res-017.pt"))
        self.image_encoder = self.visual_autoencoder.encoder

    def get_env(self, cfg):
        env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)
        return env

    @torch.no_grad()
    def get_obs(self):
        #im = obs['rgb']
       # im = obs['scene_obs']['rgb_obs']['rgb_static']
       # o = self.transform(im).unsqueeze(0).to(self.device)
         
        rgb_obs = self.env.get_obs()['rgb_obs']
       #  im = rgb_obs['rgb_static']
        im = rgb_obs['rgb_gripper']
        o = self.transform(im).unsqueeze(0).to(self.device)

        return o

    def _sample_seq(self):
        return np.random.choice(self.dataset)

    @torch.no_grad()
    def test(self, use_skill_prior=True):

        self.env.reset()


      #  obs = self.get_obs()
      #  plt.imshow(obs)
      #  plt.show()
      #  return
        steps = 0
        
        instruction = "grasp the drawer handle, then open it"

        for ep in range(100):

            while(True):

                clip_text_features = self.clip_text_encoder(clip.tokenize(instruction).to(self.device)).to(torch.float)
                text_encoding = clip_text_features
                #text_encoding = self.model.text_projection_head(clip_text_features)
                #text_encoding = text_encoding / text_encoding.norm(dim=1, keepdim=True)

                o = self.get_obs()
                o = self.image_encoder(o)
                a = self.policy(text_encoding, o).detach().cpu().numpy()[0]



                # unnormalize actions
                a = a * self.action_stats.std + self.action_stats.mean
                # change gripper action to 1 or -1
                if abs(a[-1] - 1) < abs(a[-1] - (-1)):
                    a[-1] = 1
                else:
                    a[-1] = -1
                print(np.array(a))

                # a = np.array((0., 0, 0, 0, 0, 0, 1))
        
                

                # Closed loop decoding
                obs, reward, done, info = self.env.step(a)
                # pdb.set_trace()

                # self.env.render()
                im_rgb = cv2.cvtColor(obs['rgb_obs']['rgb_gripper'], cv2.COLOR_BGR2RGB)
                cv2.imshow('w', im_rgb)
                cv2.waitKey(0)

                steps += 1

                if steps > 64 or done:

                    obs = self.env.reset()
                    o = self.get_obs(obs)
                    steps = 0
                    break


@hydra.main(config_path="/media/tim/E/calvin_env/conf", config_name="config_data_collection")
def main(cfg):
   # cfg.cameras = 'static_and_gripper'
   # print("cameras: ", cfg.cameras)
    bc_agent = EvaluateBCAgent(cfg)
    bc_agent.test()

if __name__ == "__main__":
    main()