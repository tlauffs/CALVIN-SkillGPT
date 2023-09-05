import numpy as np

from pathlib import Path
import cv2
import numpy as np
import config as CFG

def visualize():

    path = CFG.datapath_training
    data = ["rgb_static", "rgb_gripper"]

    if not Path(path).is_dir():
        print(f"Path {path} is either not a directory, or does not exist.")
        exit()

    indices = next(iter(np.load(f"{path}/scene_info.npy", allow_pickle=True).item().values()))
    indices = list(range(indices[0], indices[1] + 1))

    scene_info = np.load(f"{path}/scene_info.npy", allow_pickle=True)
    print(scene_info)

    annotations = np.load(f"{path}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
    annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
    print(annotations)
    print(len(annotations))

    # idx = 0
    idx = 60000
    ann_idx = -1

    while True:
        t = np.load(f"{path}/episode_{indices[idx]:07d}.npz", allow_pickle=True)

        for d in data:
            if d not in t:
                print(f"Data {d} cannot be found in transition")
                continue

            img = cv2.resize(t[d], (400, 400))
            cv2.imshow(d, img[:, :, ::-1])

        for n, ((low, high), ann) in enumerate(annotations):
            if indices[idx] >= low and indices[idx] <= high:
                if n != ann_idx:
                    print(f"{ann}")
                    ann_idx = n

        # user_input = input("Enter something: ")


        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == 83:  # Right arrow
            idx = (idx + 1) % len(indices)
        elif key == 81:  # Left arrow
            idx = (len(indices) + idx - 1) % len(indices)
        else:
            print(f'Unrecognized keycode "{key}"')

        