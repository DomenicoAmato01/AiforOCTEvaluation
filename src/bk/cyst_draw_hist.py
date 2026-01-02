import numpy as np
import pandas as pd
from cv2 import imread, IMREAD_GRAYSCALE
from matplotlib import pyplot as plt

def plot_patients_area_hist(df_path):

    df = pd.read_excel(df_path)

    area_0 = []
    area_1 = []
    area_2 = []
    area_3 = []
    area_4 = []
    p_names = []
    for i, fn in enumerate(df["nomefile"]):
        p_names.append(fn.split('_')[0])
    p_names = list(set(p_names))
    for p in p_names:

        df_p = df.loc[df['nomefile'].str.contains(p)]
        
        area_0 = df_p.loc[df_p['c'] == 0,'area']
        area_1 = df_p.loc[df_p['c'] == 1,'area']
        area_2 = df_p.loc[df_p['c'] == 2,'area']
        area_3 = df_p.loc[df_p['c'] == 3,'area']
        area_4 = df_p.loc[df_p['c'] == 4,'area']

        # plt.hist(area_0, bins=20)
        plt.hist(area_1, bins=10, alpha=0.5)
        plt.hist(area_2, bins=10, alpha=0.5)
        plt.hist(area_3, bins=10, alpha=0.5)
        plt.hist(area_4, bins=10, alpha=0.5)
        plt.title(f"Patient {p}")
        plt.show()

def plot_area_hist(df_path):

    df = pd.read_excel(df_path)

    area_0 = []
    eu_area_0 = []
    area_norm_0 = []
    eu_area_norm_0 = []
    num_cysts_0 = []
    area_1 = []
    eu_area_1 = []
    area_norm_1 = []
    eu_area_norm_1 = []
    num_cysts_1 = []
    area_2 = []
    eu_area_2 = []
    area_norm_2 = []
    eu_area_norm_2 = []
    num_cysts_2 = []
    area_3 = []
    eu_area_3 = []
    area_norm_3 = []
    eu_area_norm_3 = []
    num_cysts_3 = []
    area_4 = []
    eu_area_4 = []
    area_norm_4 = []
    eu_area_norm_4 = []
    num_cysts_4 = []

    for i, fn in enumerate(df["nomefile"]):
        p_name = fn.split('_')[0]

        area = df.loc[i,'area']
        eu_area = df.loc[i,'eucl_area']
        area_norm = df.loc[i,'area_norm']
        eu_area_norm = df.loc[i,'eu_area_norm']
        num_cyst = df.loc[i,'num_cysts']

        match df.loc[i,'c']:
            case 0:
                area_0.append(area)
                eu_area_0.append(eu_area)
                area_norm_0.append(area_norm)
                eu_area_norm_0.append(eu_area_norm)
                num_cysts_0.append(num_cyst)
            case 1:
                area_1.append(area)
                eu_area_1.append(eu_area)
                area_norm_1.append(area_norm)
                eu_area_norm_1.append(eu_area_norm)
                num_cysts_1.append(num_cyst)
            case 2:
                area_2.append(area)
                eu_area_2.append(eu_area)
                area_norm_2.append(area_norm)
                eu_area_norm_2.append(eu_area_norm)
                num_cysts_2.append(num_cyst)
            case 3:
                area_3.append(area)
                eu_area_3.append(eu_area)
                area_norm_3.append(area_norm)
                eu_area_norm_3.append(eu_area_norm)
                num_cysts_3.append(num_cyst)
            case 4:
                area_4.append(area)
                eu_area_4.append(eu_area)
                area_norm_4.append(area_norm)
                eu_area_norm_4.append(eu_area_norm)
                num_cysts_4.append(num_cyst)

    n_bins = 150
    #plt.hist(area_0, bins=50)
    plt.hist(area_1, bins=n_bins, alpha=0.5)
    plt.hist(area_2, bins=n_bins, alpha=0.5)
    plt.hist(area_3, bins=n_bins, alpha=0.5)
    plt.hist(area_4, bins=n_bins, alpha=0.5)
    plt.savefig("./plot/area_hist.png")
    plt.show()


    plt.hist(eu_area_1, bins=n_bins, alpha=0.5)
    plt.hist(eu_area_2, bins=n_bins, alpha=0.5)
    plt.hist(eu_area_3, bins=n_bins, alpha=0.5)
    plt.hist(eu_area_4, bins=n_bins, alpha=0.5)
    plt.savefig("./plot/eu_area_hist.png")
    plt.show()

    plt.hist(area_norm_1, bins=n_bins, alpha=0.5)
    plt.hist(area_norm_2, bins=n_bins, alpha=0.5)
    plt.hist(area_norm_3, bins=n_bins, alpha=0.5)
    plt.hist(area_norm_4, bins=n_bins, alpha=0.5)
    plt.savefig("./plot/area_norm_hist.png")
    plt.show() 

    plt.hist(eu_area_norm_1, bins=n_bins, alpha=0.5)
    plt.hist(eu_area_norm_2, bins=n_bins, alpha=0.5)    
    plt.hist(eu_area_norm_3, bins=n_bins, alpha=0.5)
    plt.hist(eu_area_norm_4, bins=n_bins, alpha=0.5)
    plt.savefig("./plot/eu_area_norm_hist.png")
    plt.show()

    plt.hist(num_cysts_1, bins=n_bins, alpha=0.5)
    plt.hist(num_cysts_2, bins=n_bins, alpha=0.5)    
    plt.hist(num_cysts_3, bins=n_bins, alpha=0.5)
    plt.hist(num_cysts_4, bins=n_bins, alpha=0.5)
    plt.savefig("./plot/num_cysts.png")
    plt.show()

if __name__ == "__main__":
    plot_area_hist("data/esaso_eval/cyst.xlsx")
    # plot_patients_area_hist("data/esaso_eval/cyst.xlsx")

    