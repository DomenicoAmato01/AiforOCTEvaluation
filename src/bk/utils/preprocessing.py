import pandas as pd
import numpy as np
from cv2 import imread, imwrite, threshold, THRESH_BINARY, IMREAD_GRAYSCALE, connectedComponents

def split_train_test(df_path):
    df = pd.read_excel(df_path)

    for i, fn in enumerate(df["nomefile"]):
        p_name = fn.split('_')[0]

        if p_name == "AMAIGN8734" or p_name == "BAGCAT3849":
            df.loc[i, 'set'] = 'test'
        else:
            df.loc[i, 'set'] = 'train'

    df.to_excel("data/esaso_eval/cyst_train_test_split.xlsx", index=False)

def mask_threshold(df_path, type_dir="Cyst", masks_dir = "masks_np", save_dir = "masks"):

    df = pd.read_excel(df_path)
    none_image = []

    for i, fn in enumerate(df["nomefile"]):
        p_name = fn.split('_')[0]

        mask_path = f"data/{type_dir}/{p_name}/{masks_dir}/{fn.split('.')[0]}m.png"
        
        mask = imread(mask_path, IMREAD_GRAYSCALE)
        if(mask is not None):
            _, mask = threshold(mask,127,255,THRESH_BINARY)
        else:
            none_image.append(mask_path)
            mask = np.zeros((512,512))

        imwrite(mask_path.replace(masks_dir,save_dir), mask)

def add_class_label(df_path):

    df = pd.read_excel(df_path)
    esaso_eval = df['!']
    labels = []
    for i, fn in enumerate(df["nomefile"]):
        
        p_name = fn.split('_')[0]

        if esaso_eval[i] == 'ok':
            labels.append(df.loc[i,'m.v.'])
        else:
            if(df.loc[i,'m.v.'] == df.loc[i,'e.c.'] or df.loc[i,'m.v.'] == df.loc[i,'a.l.f.']):
                labels.append(df.loc[i,'m.v.'])
            elif(df.loc[i,'e.c.'] == df.loc[i,'a.l.f.']):
                labels.append(df.loc[i,'e.c.'])
            else:
                labels.append(4)
    df["class"] = labels

    df.to_excel(df_path, index=False)

def add_cyst_euclidean_area(df_path, type_dir="Cyst", masks_dir = "masks"):
    df = pd.read_excel(df_path)
    none_image = []

    area = []
    euclidean_mask = []
    
    for i, fn in enumerate(df["nomefile"]):
        p_name = fn.split('_')[0]

        mask_path = f"data/{type_dir}/{p_name}/{masks_dir}/{fn.split('.')[0]}m.png"
        # print(mask_path)
        
        mask = imread(mask_path, IMREAD_GRAYSCALE)
        if(mask is not None):
            # print(mask.shape)
            mask = mask/255.0
            # print(np.unique(mask.flatten()))
        else:
            none_image.append(mask_path)
            mask = np.zeros((512,512))

        if(len(euclidean_mask) == 0):
            euclidean_mask = np.zeros_like(mask)

            center = mask.shape[0]//2
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                        euclidean_mask[i,j] = np.sqrt((i - center)**2 + (j - center)**2)
            euclidean_mask = np.max(euclidean_mask.flatten()) - euclidean_mask


        area.append(np.sum(mask*euclidean_mask))
    df["eucl_area"] = area

    print(df)
    df.to_excel(df_path, index=False)

def add_cyst_area(df_path, type_dir="Cyst", masks_dir = "masks"):

    df = pd.read_excel(df_path)
    none_image = []

    area = []
    for i, fn in enumerate(df["nomefile"]):
        p_name = fn.split('_')[0]

        mask_path = f"data/{type_dir}/{p_name}/{masks_dir}/{fn.split('.')[0]}m.png"
        # print(mask_path)
        
        mask = imread(mask_path, IMREAD_GRAYSCALE)
        if(mask is not None):
            # print(mask.shape)
            mask = mask/255.0
            # print(np.unique(mask.flatten()))
        else:
            none_image.append(mask_path)
            mask = np.zeros((512,512))

        area.append(np.sum(mask))
    df["area"] = area

    print(df)
    df.to_excel(df_path, index=False)

def add_number_of_cysts(df_path, type_dir="Cyst", masks_dir = "masks"):

    df = pd.read_excel(df_path)
    none_image = []

    num = []
    for i, fn in enumerate(df["nomefile"]):
        p_name = fn.split('_')[0]

        mask_path = f"data/{type_dir}/{p_name}/{masks_dir}/{fn.split('.')[0]}m.png"
        
        mask = imread(mask_path, IMREAD_GRAYSCALE)
        if(mask is not None):
            mask = mask/255.0
        else:
            none_image.append(mask_path)
            mask = np.zeros((512,512))

        # Count number of connected components in the mask
        num_labels, labels_im = connectedComponents((mask*255).astype(np.uint8))
        num.append(num_labels - 1)  # Subtract 1 to ignore the background label

    df["num_cysts"] = num
    df.to_excel(df_path, index=False)

def add_cyst_area_norm(df_path):

    df = pd.read_excel(df_path)
    num = []
    for i, fn in enumerate(df["nomefile"]):
        area = df.loc[i,'area']
        eu_area = df.loc[i,'eucl_area']
        num = df.loc[i,'num_cysts']

        df.loc[i,'area_norm'] = area/num if num != 0 else 0
        df.loc[i,'eu_area_norm'] = eu_area/num if num != 0 else 0
    df.to_excel(df_path, index=False)   

