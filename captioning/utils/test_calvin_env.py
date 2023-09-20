import hydra

@hydra.main(config_path="/media/tim/E/calvin_env/conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)

run_env()