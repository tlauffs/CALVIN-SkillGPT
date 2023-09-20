from pathlib import Path
import hydra
from omegaconf import OmegaConf


def get_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)
    return env


@hydra.main(config_path="/media/tim/E/calvin_env/conf", config_name="config_data_collection")
def main(cfg):
    i = get_env(cfg)
    i.reset()

if __name__ == "__main__":  
    main()