import os
import pandas as pd
from sklearn.model_selection import KFold

# grading
def Diag_Grading(IDH:str, p19q:str, His:str, CDKN:str, Grade:str)->int:
    """_summary_

    Args:
        IDH (str): IDH mutant or IDH widetype
        p19q (str): 1p/19q codel or non-codel
        His (str): histology discription
        Grade (str): grade class

    Returns:
        int: new label
    """

    if str(IDH) == "WT":
        Diag = 0  # G4 GBM
    elif str(IDH) == "Mutant":
        if str(p19q) == "codel":
            if Grade == "G2":
                Diag = 2 # G2 Oligo
            else:
                Diag = 1 # G3 Oligo


        elif str(p19q) == "non-codel":
            if His == "glioblastoma" or CDKN == "-1" or CDKN == "-2":
                Diag = 0 # G4 Astro
            else:
                if Grade == "G2":
                    Diag = 2 # G2 Astro
                elif Grade == "G3":
                    Diag = 1 # G3 Astro
                else:
                    print(f"none type!!!!!!!!!!!")
    else:
        # import pdb;pdb.set_trace()
        print(f"note!!!{IDH}")
 
    return Diag


# sub-typing
def Diag_Subtyping(IDH:str, p19q:str, His:str, CDKN:str, Grade:str)->int:
    """_summary_

    Args:
        IDH (str): IDH mutant or IDH widetype
        p19q (str): 1p/19q codel or non-codel
        His (str): histology discription
        Grade (str): grade class

    Returns:
        int: new label
    """

    if str(IDH) == "WT":
        Diag = 0  # G4 GBM
    elif str(IDH) == "Mutant":
        if str(p19q) == "codel":
            if Grade == "G2":
                Diag = 2 # G2 Oligo
            else:
                Diag = 2 # G3 Oligo

        elif str(p19q) == "non-codel":
            if His == "glioblastoma" or CDKN == "-1" or CDKN == "-2":
                Diag = 1 # G4 Astro
            else:
                if Grade == "G2":
                    Diag = 1 # G2 Astro
                elif Grade == "G3":
                    Diag = 1 # G3 Astro
                else:
                    print(f"none type!!!!!!!!!!!")
    else:
        # import pdb;pdb.set_trace()
        print(f"note!!!{IDH}")
 
    return Diag

def Diag_Grading6(IDH:str, p19q:str, His:str, CDKN:str, Grade:str)->int:
    """_summary_

    Args:
        IDH (str): IDH mutant or IDH widetype
        p19q (str): 1p/19q codel or non-codel
        His (str): histology discription
        Grade (str): grade class

    Returns:
        int: new label
    """

    if str(IDH) == "WT":
        Diag = 0  # G4 GBM
    elif str(IDH) == "Mutant":
        if str(p19q) == "codel":
            if Grade == "G2":
                Diag = 5 # G2 Oligo
            else:
                Diag = 4 # G3 Oligo
  
        elif str(p19q) == "non-codel":
            if His == "glioblastoma" or CDKN == "-1" or CDKN == "-2":
                Diag = 1 # G4 Astro
            else:
                if Grade == "G2":
                    Diag = 3 # G2 Astro
                elif Grade == "G3":
                    Diag = 2 # G3 Astro
                else:
                    print(f"none type!!!!!!!!!!!")
    else:
        # import pdb;pdb.set_trace()
        print(f"note!!!{IDH}")
 
    return Diag




def Diag_Grading4(IDH:str, p19q:str, His:str, CDKN:str, Grade:str)->int:
    """_summary_

    Args:
        IDH (str): IDH mutant or IDH widetype
        p19q (str): 1p/19q codel or non-codel
        His (str): histology discription
        Grade (str): grade class

    Returns:
        int: new label
    """

    if str(IDH) == "WT":
        Diag = 0  # G4 GBM
    elif str(IDH) == "Mutant":
        if str(p19q) == "codel":
            if Grade == "G2":
                Diag = 3 # G2 Oligo
            else:
                Diag = 3 # G3 Oligo
  
        elif str(p19q) == "non-codel":
            if His == "glioblastoma" or CDKN == "-1" or CDKN == "-2":
                Diag = 1 # G4 Astro
            else:
                if Grade == "G2":
                    Diag = 2 # G2 Astro
                elif Grade == "G3":
                    Diag = 2 # G3 Astro
                else:
                    print(f"none type!!!!!!!!!!!")
    else:
        # import pdb;pdb.set_trace()
        print(f"note!!!{IDH}")
 
    return Diag


def Survival_T(patients_df, n_bins: int=4, eps=1e-6):
    """_summary_

    Args:
        IDH (str): IDH mutant or IDH widetype
        p19q (str): 1p/19q codel or non-codel
        His (str): histology discription
        Grade (str): grade class

    Returns:
        int: new label
    """
    days_in_a_month = 30.44  # 平均一个月的天数
    
    patients_df['survival_months'] = patients_df['OS.time'].apply(lambda x: round(x / days_in_a_month, 2))
    patients_df.replace('#N/A', pd.NA, inplace=True)

    # 删除包含 NaN 的行
    patients_df.dropna(subset=['OS.time'], inplace=True)
    patients_df['OS'].replace({0: 1, 1: 0}, inplace=True)
    # import pdb;pdb.set_trace()
    uncensored_df = patients_df[patients_df['OS'] == 0]
    
    disc_labels, q_bins = pd.qcut(uncensored_df['survival_months'], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = patients_df['survival_months'].max() + eps
    q_bins[0] = patients_df['survival_months'].min() - eps

    disc_labels, q_bins = pd.cut(patients_df['survival_months'], bins=q_bins, retbins=True,
    labels=False, right=False, include_lowest=True)
    
    surv_df = pd.DataFrame()
    surv_df.insert(0, 'patients', patients_df["bcr_patient_barcode"].values)
    surv_df.insert(1, 'labels', disc_labels.values.astype(int))
    surv_df.insert(2, 'survival_months', patients_df['survival_months'].values)
    surv_df.insert(3, 'censorship', patients_df['OS'].values.astype(int))

    return surv_df


if __name__ == "__main__":

    type = "survival"  # 1. grading  2. subtyping 3.survival 4.classification
    path_dir = "DATASET/tcga_glioma"
    bag_path = "DATASET/tcga_glioma/features_clip_vit_b16"
    mol_path = "DATASET/tcga_glioma/molecular"
        
    tabular = "DATASET/tcga_glioma/labels/TCGA_patientLevel.xlsx"
    
    os_update = "DATASET/tcga_glioma/labels/updated_OS.xlsx"
    
    out_dir = os.path.join(path_dir, "labels", type)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    
    ins_names = os.listdir(bag_path)
    mol_names = os.listdir(mol_path)
    
    bag_names = []
    
    
    for mol_name in mol_names:
        ins_name = mol_name.replace('.csv', '.h5')
        if ins_name in ins_names:
            bag_names.append(ins_name)
    
    
    # print(bag_names)
    # 读取 CSV 文件
    df = pd.read_excel(tabular)
    os_df = pd.read_excel(os_update)
    df = df[df['Patient ID'].isin(os_df['bcr_patient_barcode'])]

    Patient_Diags = [] 
    # print(df.keys())
    
    if type == "survival":
        
        newdf = Survival_T(patients_df=os_df)
        newdf = newdf[newdf['patients'].isin(df['Patient ID'])]
        for index, row in newdf.iterrows():
            patients = row["patients"]
            labels = row["labels"]
            survival_months = row["survival_months"]
            censorship = row["censorship"]
            sub_diag =  {patients: [labels, survival_months, censorship]}
            Patient_Diags.append(sub_diag)
        print(newdf["labels"].value_counts())
    else:
        new_Diags = dict()
        for index, row in df.iterrows():
            # print(f"Index: {index}, Histology: {row['Histology']}, Grade: {row['Grade']}")
            if str(row['IDH status']) == 'nan' and str(row['1p/19q codeletion']) == 'nan' and str(row['Histology']) == 'nan' and \
                str(row['2016-Grade']) == 'nan':
                continue
            ## grading
            if type == "grading":
                new_label = Diag_Grading(IDH=row['IDH status'], p19q=row['1p/19q codeletion'], His=row['Histology'], CDKN=row['CDKN2A'], Grade=row['2016-Grade'])
            elif type == "subtyping":
                new_label = Diag_Subtyping(IDH=row['IDH status'], p19q=row['1p/19q codeletion'], His=row['Histology'], CDKN=row['CDKN2A'], Grade=row['2016-Grade'])
            elif type == "classification":
                new_label = Diag_Grading6(IDH=row['IDH status'], p19q=row['1p/19q codeletion'], His=row['Histology'], CDKN=row['CDKN2A'], Grade=row['2016-Grade'])   
            else:
                raise NotImplementedError
        
            print(f"Index: {index}, New Label: {new_label}")
            Patient_ID = str(row['Patient ID'])
            sub_diag =  {Patient_ID: new_label}
            
            print(f"Patient_ID: {Patient_ID},  Sub_diag: {sub_diag}")
            new_Diags.update(sub_diag)
            Patient_Diags.append(sub_diag)

        print(len(Patient_Diags))

        # import pdb;pdb.set_trace()

        # 将字典转换为 DataFrame
        patient_df = pd.DataFrame(list(new_Diags.items()), columns=['patients',  'labels'])

        path = f'{out_dir}/{type}_patient.csv'
        patient_df.to_csv(path, index=False)
        name = path.split("/")[-1]
        print(f"Patient CSV file has been created as '{name}'")

        df = pd.read_csv(path)
        print(df["labels"].value_counts())
        
        
    # import pdb;pdb.set_trace()
        
    
    # 创建5折交叉验证的KFold对象
    kf = KFold(n_splits=5, shuffle=True, random_state=43)



    index=1
    # 执行5折交叉验证
    for train_indexs, test_indexs in kf.split(Patient_Diags):
        # import pdb;pdb.set_trace()
        train_path = os.path.join(out_dir, f"{type}_train_{index}.csv")
        train_name = train_path.split("/")[-1]
        
        test_path = os.path.join(out_dir, f"{type}_test_{index}.csv")
        test_name = test_path.split("/")[-1]
        
        train_Diags = dict()
        for train_index in train_indexs:
            Patient_Diag = Patient_Diags[train_index]
            Patient_name = list(Patient_Diag.keys())[0]
            new_label =  list(Patient_Diag.values())[0]
            WSI_names = [bag_name for bag_name in bag_names if Patient_name in bag_name]
            WSI_diag =  {WSI_name: new_label for WSI_name in WSI_names}
            train_Diags.update(WSI_diag)
             
        test_Diags = dict()     
        for test_index in test_indexs:
            Patient_Diag = Patient_Diags[test_index]
            Patient_name = list(Patient_Diag.keys())[0]
            new_label =  list(Patient_Diag.values())[0]
            WSI_names = [bag_name for bag_name in bag_names if Patient_name in bag_name]
            WSI_diag =  {WSI_name: new_label for WSI_name in WSI_names}
            test_Diags.update(WSI_diag) 
            
                   
            
        if type != "survival": 
            train_WSI_df = pd.DataFrame(list(train_Diags.items()), columns=['features',  'labels'])   
            test_WSI_df = pd.DataFrame(list(test_Diags.items()), columns=['features',  'labels'])
        else:  
            train_list =   [] 
            for train_Diag in train_Diags.items():
                Diag = [train_Diag[0]] + train_Diag[1]                
                train_list.append(Diag)
            train_WSI_df = pd.DataFrame(train_list, columns=['features', 'labels', 'survival_months', "censorship"])
            
            
            test_list =   [] 
            for test_Diag in test_Diags.items():
                Diag = [test_Diag[0]] + test_Diag[1]                
                test_list.append(Diag)
            test_WSI_df = pd.DataFrame(test_list, columns=['features', 'labels', 'survival_months', "censorship"])
                      
        
        
        train_WSI_df.to_csv(train_path, index=False)
        print(f"fold {index} CSV file has been created as '{train_path}'")
        train_df = pd.read_csv(train_path)
        print(train_df["labels"].value_counts())       
        
        
        test_WSI_df.to_csv(test_path, index=False)
        print(f"fold {index} CSV file has been created as '{test_path}'")
        test_df = pd.read_csv(test_path)
        print(test_df["labels"].value_counts())
        
        index +=  1
