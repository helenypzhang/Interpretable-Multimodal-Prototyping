import json
import os
import pandas as pd


def load_json(filenamejson):
    with open(filenamejson) as f:
        raw_data = json.load(f)
    return raw_data


if __name__ == "__main__":
    
    json_path = "DATASET/tcga_glioma/labels/metadata.cart.2023-10-29.json"
    bomarker_path = "DATASET/tcga_glioma/labels/TCGA_GBM_gene_sorted.csv"
    signature = "DATASET/tcga_glioma/labels/signatures.csv"
    oringin_dir = "DATASET/tcga_glioma/origin_molecular"
    featue_dir = "DATASET/tcga_glioma/features_r50"
    out_dir = "DATASET/tcga_glioma/molecular"
    
    
    gene_df = pd.read_csv(bomarker_path)
    gene_df = gene_df.sort_values(by='variance', ascending=False).head(1039)
    
    
    

    # gene_df = pd.read_excel(bomarker_path)
    
    
    
    # both_list = gene_df[gene_df['Type'] == 'Both']["gene_symbol"].dropna().tolist()  # 27
    # tumor_list = gene_df[gene_df['Type'] == 'Tumor']["gene_symbol"].dropna().tolist() # 113
    # immune_list = gene_df[gene_df['Type'] == 'Immune']["gene_symbol"].dropna().tolist() # 1645
    
    gene_list = pd.read_csv(signature)
    tumor_list = gene_list["Tumor Suppressor Genes"].dropna().tolist()
    oncogenes_list = gene_list["Oncogenes"].dropna().tolist()
    protein_list = gene_list["Protein Kinases"].dropna().tolist()
    cell_list = gene_list["Cell Differentiation Markers"].dropna().tolist()
    transcription_list = gene_list["Transcription Factors"].dropna().tolist()
    cytokines_list = gene_list["Cytokines and Growth Factors"].dropna().tolist()
    
    
    # candidate_list = tumor_list + oncogenes_list + protein_list + cell_list + transcription_list + cytokines_list
    
    # tumor_df = gene_df[gene_df["gene_name"].isin(tumor_list)]
    # oncogenes_df = gene_df[gene_df["gene_name"].isin(oncogenes_list)]
    # protein_df = gene_df[gene_df["gene_name"].isin(protein_list)]
    # cell_df = gene_df[gene_df["gene_name"].isin(cell_list)]
    # transcription_df = gene_df[gene_df["gene_name"].isin(transcription_list)]
    # cytokines_df = gene_df[gene_df["gene_name"].isin(cytokines_list)]
    # import pdb;pdb.set_trace()
    
    
    c_gene_list = gene_df["gene_name"].dropna().tolist()
    
    

    WSI_IDs = os.listdir(os.path.join(featue_dir))
    
    json_file = load_json(json_path)
    count = 0
    for file in json_file:
        file_id = file['file_id']
        file_name = file['file_name']
        subtrings = file['associated_entities'][0]['entity_submitter_id'].split("-")[:3]
        new_entity_id = "-".join(subtrings)
        file_path = os.path.join(oringin_dir, file_id, file_name)
        df = pd.read_csv(file_path, sep='\t', skiprows=1) 
        context = df[df['gene_name'].isin(c_gene_list)]  # 1790
        # both_context = df[df['gene_name'].isin(both_list)] # 27
        # tumor_context = df[df['gene_name'].isin(tumor_list)]  # 113
        # immune_context = df[df['gene_name'].isin(immune_list)] # 1650
        
        tumor_context = df[df['gene_name'].isin(tumor_list)]
        oncogenes_context = df[df['gene_name'].isin(oncogenes_list)]
        protein_context = df[df['gene_name'].isin(protein_list)]
        cell_context = df[df['gene_name'].isin(cell_list)]
        transcription_context = df[df['gene_name'].isin(transcription_list)]
        cytokines_context = df[df['gene_name'].isin(cytokines_list)]
        print(f"total Genes: {len(context)} \n ")
        print(f"total Tumor Suppressor Genes: {len(tumor_context)} \n \
              Oncogenes: {len(oncogenes_context)} \n \
              Protein Kinases:   {len(protein_context)} \n \
              Cell Differentiation Markers: {len(cell_list)} \n \
              Transcription Factors: {len(transcription_context)} \n \
              Cytokines and Growth Factors: {len(cytokines_context)} \n \
                  ")
        # import pdb;pdb.set_trace()
        
        # import pdb;pdb.set_trace()
        # sub_WSI_IDs = [WSI_ID.replace(".h5", ".csv") for WSI_ID in WSI_IDs if new_entity_id in WSI_ID]
        # for WSI_ID in sub_WSI_IDs:
        #     file_name = WSI_ID
        #     path = os.path.join(out_dir, file_name)
        #     # import pdb;pdb.set_trace()
        #     context.to_csv(path, index=False)
        #     count += 1
        #     print(f"CSV file has been created as '{file_name}'")
    print(f"total file num: {count}")
