import numpy as np
import clip
import torch
from dataset import AttrDict 
from torchvision import transforms
from behavior_encoder import BehaviourEncoder
from bc_policy import LanguageConditionedPolicy
from visual_autoencoder import VisualAutoencoder
import config as CFG
import pdb
import cv2



def _convert_image_to_rgb(image):
    return image.convert("RGB")

def clip_preprocess(n_px=224):
    return Compose([
            ToPILImage(mode='RGB'),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])


class EvaluateBCAgent():
    def __init__(self):
        
        self.device = CFG.device

       
        self.env = language_table.LanguageTable(
                        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
                        reward_factory=block2block.BlockToBlockReward,
                        control_frequency=20.0,
                    )
     

        clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=True)
        self.clip_text_encoder = clip_model.encode_text
        self.clip_image_encoder = clip_model.encode_image
        #self.transform = clip_preprocess()

        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((CFG.img_size)),
            transforms.ToTensor() 
        ])

        self.model = BehaviourEncoder(temperature=0.07).to(CFG.device)
        self.model.load_state_dict(torch.load("/home/krishan/work/skillGPT/skillGPT/checkpoints/lang_table_sim_full_clip_encoding_fixed_temp_softmax_gt_2023.03.16_16:04:39_alignment_best_MARCH_BEST.pth"))
        self.model.eval()
        self.model.to(self.device)

        self.policy = LanguageConditionedPolicy(language_dim=512, action_dim=2)
        self.policy.load_state_dict(torch.load("/home/krishan/work/skillGPT/skillGPT/checkpoints/bc_policy/bc_policy.pt"))
        self.policy = self.policy.to(self.device)

        self.action_stats = AttrDict(mean=np.array([0.0001016613095998764, 0.0026315886061638594], dtype=np.float32),
                                     std=np.array([0.008755197748541832, 0.010186241939663887]), dtype=np.float32)
        
        # initialize the visual autoencoder
        self.visual_autoencoder = VisualAutoencoder(512)
        self.visual_autoencoder = self.visual_autoencoder.to(self.device)
        
        # load the weights
        self.visual_autoencoder.load_state_dict(torch.load("/home/krishan/work/skillGPT/skillGPT/checkpoints/image_vae_high_res.pth"))
        self.image_encoder = self.visual_autoencoder.encoder



    @torch.no_grad()
    def get_obs(self, obs):
        im = obs['rgb']
        o = self.transform(im).unsqueeze(0).to(self.device)
        #o = self.clip_image_encoder(self.transform(im).unsqueeze(0).to(self.device)).detach().to(torch.float)
        return o

    def _sample_seq(self):
        return np.random.choice(self.dataset)

    @torch.no_grad()
    def test(self, use_skill_prior=True):

        obs = self.env.reset()
        pdb.set_trace()
        steps = 0
        
        instruction = "move the blue moon close to the red moon"

        for ep in range(100):
            while(True):
                clip_text_features = self.clip_text_encoder(clip.tokenize(instruction).to(self.device)).to(torch.float)
                text_encoding = clip_text_features
                #text_encoding = self.model.text_projection_head(clip_text_features)
                #text_encoding = text_encoding / text_encoding.norm(dim=1, keepdim=True)

                o = self.get_obs(obs)
                o = self.image_encoder(o)
                a = self.policy(text_encoding, o).detach().cpu().numpy()[0]

                print(a)

                # unnormalize actions
                a = a * self.action_stats.std + self.action_stats.mean

                # Closed loop decoding
                obs, reward, done, _ = self.env.step(a)

                # self.env.render()
                im_rgb = cv2.cvtColor(obs['rgb'], cv2.COLOR_BGR2RGB)
                cv2.imshow('w', im_rgb)
                cv2.waitKey(10)

                steps += 1

                if steps > 45 or done:
                    obs = self.env.reset()
                    o = self.get_obs(obs)
                    steps = 0
                    break


