import torch
import clip
import os
import re
import argparse
import h5py


import os.path as osp
from PIL import Image

from tqdm import tqdm

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}



def main(args):
    output_dir = os.path.join(args.output_dir, args.model, args.dataset)
    root = os.path.join(args.root, args.dataset, "images/train")
    if args.model == "clip_vit_b32":
        model_flag = "ViT-B/32"
    elif args.model == "clip_vit_b16":
        model_flag = "ViT-B/16"
    elif args.model == "clip_vit_l14":
        model_flag = "ViT-L/14"
    elif args.model == "clip_vit_l14@336px":
        model_flag = "ViT-L/14@336px"
    elif args.model == "clip_r50":
        model_flag = "RN50"
    elif args.model == "clip_r101":
        model_flag = "RN101"
    elif args.model == "clip_r50x4":
        model_flag = "RN50x4"
    elif args.model == "clip_r50x16":
        model_flag = "RN50x16"
    elif args.model == "clip_r50x64":
        model_flag = "RN50x64"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_flag, device=device)
    
    # output_dir = args.output_dir
    
    out_patchs = [path_name.split(".")[0] for path_name in os.listdir(output_dir)]

    sub_folders = os.listdir(root)
    len_folders = len(sub_folders)
    with tqdm(total=len_folders, desc="Progress 1") as pbar1:
        for sub_folder in sub_folders:
            if sub_folder not in out_patchs:
                patches = os.listdir(osp.join(root, sub_folder))
                patches = sorted(patches)

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
                        hdf5_file.create_dataset("clip_feature", data=wsi_feature)
                    print(f"Finish WSI: {sub_folder}.h5")  
                
            pbar1.update(1)


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data_coop", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="imagenet", help="dataset")
    parser.add_argument("--model", type=str, default="clip_vit_b32", help="dataset")
    parser.add_argument(
        "--output-dir", type=str, default="data_coop/feature", help="output directory"
    )

    args = parser.parse_args()

    main(args)