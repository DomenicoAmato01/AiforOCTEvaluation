import numpy as np
import pandas as pd
from cv2 import imread, IMREAD_GRAYSCALE
from matplotlib import pyplot as plt

def check_error(df_path):

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

        error = df_p.loc[(df['area'] == 0) & (df['class'] != 0)]

        print(error['nomefile'])


if __name__ == "__main__":
    check_error("data/esaso_eval/cyst.xlsx")
    # plot_patients_area_hist("data/esaso_eval/cyst.xlsx")

    