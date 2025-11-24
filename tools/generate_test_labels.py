import os
import pandas as pd

def Survival_T(patients_df, n_bins: int=4, eps=1e-6, out_path="DATASET/test/labels/survival/survival_test.csv"):
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
    
    patients_df['survival_months'] = patients_df['OS'].apply(lambda x: round(x / days_in_a_month, 2))
    patients_df.replace('#N/A', pd.NA, inplace=True)

    # 删除包含 NaN 的行
    patients_df.dropna(subset=['OS'], inplace=True)
    patients_df['event'].replace({0: 1, 1: 0}, inplace=True)
    # import pdb;pdb.set_trace()
    uncensored_df = patients_df[patients_df['event'] == 0]
    
    disc_labels, q_bins = pd.qcut(uncensored_df['survival_months'], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = patients_df['survival_months'].max() + eps
    q_bins[0] = patients_df['survival_months'].min() - eps

    # 根据得到的边界对所有数据进行分箱
    disc_labels, q_bins = pd.cut(
        patients_df['survival_months'],
        bins=q_bins,
        retbins=True,
        labels=False,
        right=False,
        include_lowest=True
    )

    # 构建输出 DataFrame
    surv_df = pd.DataFrame({
        'patients': patients_df["WSI_ID"].values,
        'labels': disc_labels.astype(int),
        'survival_months': patients_df['survival_months'].values,
        'censorship': patients_df['event'].values.astype(int)
    })

    # 保存到指定路径
    surv_df.to_csv(out_path, index=False)

    return surv_df

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

if __name__ == "__main__":
    path = "DATASET/test/labels/survival/CPTAC.xlsx"
    df = pd.read_excel(path)

    # survival
    out_path = "DATASET/test/labels/survival/survival_test.csv"
    if not os.path.exists("DATASET/test/labels/survival"):
        os.makedirs("DATASET/test/labels/survival")
    Survival_T(df)

    # grading
    out_path = "DATASET/test/labels/grading/grading_test.csv"
    if not os.path.exists("DATASET/test/labels/grading"):
        os.makedirs("DATASET/test/labels/grading")
    df = pd.read_excel(path)
    patients = []
    labels = []
    for index, row in df.iterrows():
        label = Diag_Grading(IDH=row['IDH status'], p19q=row['1p/19q codeletion'], His=row['Histology'], CDKN=row['CDKN2A'], Grade=row['2016-Grade'])
        patients.append(row['WSI_ID'])
        labels.append(label)
    out_df = pd.DataFrame({'patients': patients, 'labels': labels})
    out_df.to_csv(out_path, index=False)

    # classification
    out_path = "DATASET/test/labels/classification/classification_test.csv"
    if not os.path.exists("DATASET/test/labels/classification"):
        os.makedirs("DATASET/test/labels/classification")
    df = pd.read_excel(path)
    patients = []
    labels = []
    for index, row in df.iterrows():
        label = Diag_Grading6(IDH=row['IDH status'], p19q=row['1p/19q codeletion'], His=row['Histology'], CDKN=row['CDKN2A'], Grade=row['2016-Grade'])
        patients.append(row['WSI_ID'])
        labels.append(label)
    out_df = pd.DataFrame({'patients': patients, 'labels': labels})
    out_df.to_csv(out_path, index=False)
