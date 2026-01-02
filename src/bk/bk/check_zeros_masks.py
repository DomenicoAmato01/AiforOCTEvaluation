import numpy as np
import pandas as pd
from cv2 import imread, IMREAD_GRAYSCALE, threshold, imwrite, THRESH_BINARY
from matplotlib import pyplot as plt


def mask_threshold(df_path):

    df = pd.read_excel(df_path)
    none_image = []
    error = []
    for i, fn in enumerate(df["nomefile"]):
        p_name = fn.split('_')[0]

        mask_path = f"data/Cyst/{p_name}/labels/{fn.split('.')[0]}m.png"
        
        mask = imread(mask_path, IMREAD_GRAYSCALE)
        if(mask is None):
            m_name = mask_path.split('/')[-1]
            error.append(m_name)

            df.loc[i,'missing_mask'] = 1
        else:
            df.loc[i,'missing_mask'] = 0

    df.to_excel(df_path, index=False)
    print(error)


if __name__ == "__main__":
    mask_threshold("data/esaso_eval/cyst.xlsx")