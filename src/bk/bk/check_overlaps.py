import numpy as np
import pandas as pd
from cv2 import imread, IMREAD_GRAYSCALE, imshow, waitKey, destroyAllWindows, imwrite
from matplotlib import pyplot as plt

def count_overlaps(df_path):
    df = pd.read_excel(df_path)

    df_0 = df[df["class"]==0]
    df_1 = df[df["class"]==1]
    df_1 = df_1[df_1['area'] != 0]
    df_2 = df[df["class"]==2]
    df_2 = df_2[df_2['area'] != 0]
    df_3 = df[df["class"]==3]
    df_3 = df_3[df_3['area'] != 0]

    min_area_0 = df_0[df_0['area'] == df_0['area'].min()]
    max_area_0 = df_0[df_0['area'] == df_0['area'].max()]

    min_area_1 = df_1[df_1['area'] == df_1['area'].min()]
    max_area_1 = df_1[df_1['area'] == df_1['area'].max()]

    min_area_2 = df_2[df_2['area'] == df_2['area'].min()]
    max_area_2 = df_2[df_2['area'] == df_2['area'].max()]

    min_area_3 = df_3[df_3['area'] == df_3['area'].min()]
    max_area_3 = df_3[df_3['area'] == df_3['area'].max()]

    # print(f"Classe 0 compresa tra {min_area_0['area'].values[0]} e {max_area_0['area'].values[0]}")
    # print(f"Classe 1 compresa tra {min_area_1['area'].values[0]} e {max_area_1['area'].values[0]}")
    # print(f"Classe 2 compresa tra {min_area_2['area'].values[0]} e {max_area_2['area'].values[0]}")
    # print(f"Classe 3 compresa tra {min_area_3['area'].values[0]} e {max_area_3['area'].values[0]}")
 
    overlaps_0_1 = df_0[df_0['area']<max_area_0['area'].values[0]]
    overlaps_0_1 = overlaps_0_1[overlaps_0_1['area']>min_area_1['area'].values[0]]
    
    print(f"Class 0 overlaps with Class 1: {overlaps_0_1['nomefile'].count()}")

    overlaps_0_2 = df_0[df_0['area']<max_area_2['area'].values[0]]
    overlaps_0_2 = overlaps_0_2[overlaps_0_2['area']>min_area_2['area'].values[0]]
    
    print(f"Class 0 overlaps with Class 2: {overlaps_0_2['nomefile'].count()}")

    overlaps_0_3 = df_0[df_0['area']<max_area_3['area'].values[0]]
    overlaps_0_3 = overlaps_0_3[overlaps_0_3['area']>min_area_3['area'].values[0]]
    
    print(f"Class 0 overlaps with Class 3: {overlaps_0_3['nomefile'].count()}")

    number_0 = df_0['nomefile'].count()

    print(f"Class 0 items: {number_0}")

    '''Class 1'''
    overlaps_1_0 = df_1[df_1['area']<max_area_0['area'].values[0]]
    overlaps_1_0 = overlaps_1_0[overlaps_1_0['area']>min_area_0['area'].values[0]]
    
    print(f"Class 1 overlaps with Class 0: {overlaps_1_0['nomefile'].count()}")

    overlaps_1_2 = df_1[df_1['area']<max_area_2['area'].values[0]]
    overlaps_1_2 = overlaps_1_2[overlaps_1_2['area']>min_area_2['area'].values[0]]
    
    print(f"Class 1 overlaps with Class 2: {overlaps_1_2['nomefile'].count()}")

    overlaps_1_3 = df_1[df_1['area']<max_area_3['area'].values[0]]
    overlaps_1_3 = overlaps_1_3[overlaps_1_3['area']>min_area_3['area'].values[0]]
    
    print(f"Class 1 overlaps with Class 3: {overlaps_1_3['nomefile'].count()}")

    number_1 = df_1['nomefile'].count()

    print(f"Class 1 items: {number_1}")

    '''Class 2'''
    overlaps_2_0 = df_2[df_2['area']<max_area_0['area'].values[0]]
    overlaps_2_0 = overlaps_2_0[overlaps_2_0['area']>min_area_0['area'].values[0]]
    
    print(f"Class 2 overlaps with Class 0: {overlaps_2_0['nomefile'].count()}")

    overlaps_2_1 = df_2[df_2['area']<max_area_1['area'].values[0]]
    overlaps_2_1 = overlaps_2_1[overlaps_2_1['area']>min_area_1['area'].values[0]]
    
    print(f"Class 2 overlaps with Class 1: {overlaps_2_1['nomefile'].count()}")

    overlaps_2_3 = df_2[df_2['area']<max_area_3['area'].values[0]]
    overlaps_2_3 = overlaps_2_3[overlaps_2_3['area']>min_area_3['area'].values[0]]
    
    print(f"Class 2 overlaps with Class 3: {overlaps_2_3['nomefile'].count()}")

    number_2 = df_2['nomefile'].count()

    print(f"Class 2 items: {number_2}")

    '''Class 3'''
    overlaps_3_0 = df_3[df_3['area']<max_area_0['area'].values[0]]
    overlaps_3_0 = overlaps_3_0[overlaps_3_0['area']>min_area_0['area'].values[0]]
    
    print(f"Class 3 overlaps with Class 0: {overlaps_3_0['nomefile'].count()}")

    overlaps_3_1 = df_3[df_3['area']<max_area_1['area'].values[0]]
    overlaps_3_1 = overlaps_3_1[overlaps_3_1['area']>min_area_1['area'].values[0]]
    
    print(f"Class 3 overlaps with Class 1: {overlaps_3_1['nomefile'].count()}")

    overlaps_3_2 = df_3[df_3['area']<max_area_2['area'].values[0]]
    overlaps_3_2 = overlaps_3_2[overlaps_3_2['area']>min_area_2['area'].values[0]]
    
    print(f"Class 3 overlaps with Class 2: {overlaps_3_2['nomefile'].count()}")

    number_3 = df_3['nomefile'].count()

    print(f"Class 3 items: {number_3}")
    


if __name__ == "__main__":
    count_overlaps("data/esaso_eval/cyst.xlsx")
    # plot_patients_area_hist("data/esaso_eval/cyst.xlsx")

    