import torch
import sys
import os
sys.path.append(os.getcwd())
from clip import clip
import re
import argparse
import h5py


import os.path as osp
from PIL import Image

from tqdm import tqdm


# 定义一个函数来从文件名中提取数字
def custom_sort(filename):
    parts = filename.split("_")
    num1 = int(parts[0])
    num2 = int(parts[1].split(".")[0])  # 去除文件扩展名后的数字部分
    return (num1, num2)

def main(args):    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    root = args.root
    output_dir = args.output_dir
    
    out_patchs = [path_name.split(".")[0] for path_name in os.listdir(output_dir)]

    sub_folders = os.listdir(root)
    len_folders = len(sub_folders)
    with tqdm(total=len_folders, desc="Progress 1") as pbar1:
        for sub_folder in sub_folders:
            if sub_folder not in out_patchs:
                patches = os.listdir(osp.join(root, sub_folder))
                patches = sorted(patches, key=custom_sort)

                len_patches = len(patches)
                patchs_list = []
                
                with tqdm(total=len_patches, desc="Progress 2", leave=False, position=1) as pbar2:
                    for patch in patches:
                        patch_path = osp.join(root, sub_folder, patch)
                        
                        patch = preprocess(Image.open(patch_path)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            patch_feature = model.encode_image(patch)
                            patchs_list.append(patch_feature)
                        pbar2.update(1)
                        
                if len(patchs_list) > 1:       
                    wsi_feature = torch.cat(patchs_list, dim=0)    
                    # import pdb;pdb.set_trace()
                    wsi_path = osp.join(output_dir, f"{sub_folder}.h5")
                    # 创建HDF5文件并写入数据
                    with h5py.File(wsi_path, "w") as hdf5_file:
                        # 将PyTorch张量转换为NumPy数组
                        wsi_feature = wsi_feature.cpu().numpy()
                        # 创建一个HDF5数据集并写入数据
                        hdf5_file.create_dataset("clip_vit_b32_feature", data=wsi_feature)
                    print(f"Finish WSI: {sub_folder}.h5")  
                
            pbar1.update(1)


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument(
        "--output-dir", type=str, default="DATASET/clip_vit_b32_feature", help="output directory"
    )

    args = parser.parse_args()

    main(args)