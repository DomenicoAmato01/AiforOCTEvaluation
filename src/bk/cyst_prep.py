from utils.preprocessing import mask_threshold, split_train_test
from utils.preprocessing import add_cyst_area, add_cyst_euclidean_area, add_number_of_cysts
from utils.preprocessing import add_cyst_area_norm

if __name__ == "__main__":
    mask_threshold("data/esaso_eval/cyst.xlsx", type_dir="Cyst", masks_dir="masks_np", save_dir="masks")
    add_cyst_area("data/esaso_eval/cyst.xlsx")
    add_cyst_euclidean_area("data/esaso_eval/cyst.xlsx")
    add_number_of_cysts("data/esaso_eval/cyst.xlsx")
    add_cyst_area_norm("data/esaso_eval/cyst.xlsx")
    split_train_test("data/esaso_eval/cyst.xlsx")


