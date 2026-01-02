import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def count_class_for_patients(class_couple):
    patient_counting = {}
    for fn, label in class_couple:
        patient_id = fn.split('_')[0]  # Assuming patient ID is the prefix before the first underscore
        if patient_id not in patient_counting:
            patient_counting[patient_id] = {0: 0, 1: 0, 2:0, 3:0}  # Initialize counts for classes 0, 1, 2, and 3
        patient_counting[patient_id][label] += 1
    return patient_counting

def plot_patient_class_distribution(df):
    
    class_couple = zip(df["nomefile"].to_numpy(), df["c"].to_numpy())

    patient_counts = count_class_for_patients(class_couple)

    print("Patient-wise class distribution:")
    for patient_id, counts in patient_counts.items():
        print(f"Patient ID: {patient_id}, Class 0: {counts[0]}, Class 1: {counts[1]}, Class 2: {counts[2]}, Class 3: {counts[3]}")

    # Calculate percentages for each patient
    patient_percentages = {}
    for patient_id, counts in patient_counts.items():
        total_images = sum(counts.values())
        patient_percentages[patient_id] = {
            class_label: (count / total_images) * 100 if total_images > 0 else 0
            for class_label, count in counts.items()
        }

    print("\nPatient-wise class percentages:")
    for patient_id, percentages in patient_percentages.items():
        print(f"Patient ID: {patient_id}, "
              f"Class 0: {percentages[0]:.2f}%, "
              f"Class 1: {percentages[1]:.2f}%, "
              f"Class 2: {percentages[2]:.2f}%, "
              f"Class 3: {percentages[3]:.2f}%")
    
    # Plotting grouped bar chart
    patient_ids = list(patient_percentages.keys())
    class_labels = [0, 1, 2, 3]
    
    # Prepare data for plotting
    data_to_plot = {class_label: [patient_percentages[p_id][class_label] for p_id in patient_ids] for class_label in class_labels}

    x = np.arange(len(patient_ids))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, class_label in enumerate(class_labels):
        offset = width * i
        rects = ax.bar(x + offset - (width * (len(class_labels) - 1) / 2), data_to_plot[class_label], width, label=f'Class {class_label}')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Percentage of Images per Class for Each Patient')
    ax.set_xticks(x)
    ax.set_xticklabels(patient_ids, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    
    plt.savefig("plot/patient_class_distribution.png")
    plt.show()

    
def plot_total_class_distribution(df):
    class_counts = df['c'].value_counts().sort_index()
    
    total_images = class_counts.sum()
    class_percentages = (class_counts / total_images) * 100

    print("\nTotal class percentages:")
    for class_label, percentage in class_percentages.items():
        print(f"Class {class_label}: {percentage:.2f}%")

    plt.figure(figsize=(8, 6))
    class_percentages.plot(kind='bar', color='skyblue')
    plt.title('Total Percentage of Images per Class')
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot/total_class_distribution.png")
    plt.show()

def print_analysis(df):
    class_counts = df['c'].value_counts().sort_index()
    total_images = class_counts.sum()
    class_percentages = (class_counts / total_images) * 100

    print(f"\nTotal number of images: {total_images}")
    patient_counts = count_class_for_patients(zip(df["nomefile"].to_numpy(), df["c"].to_numpy()))
    print("\n Distribution of images per patient:")
    for patient_id, counts in patient_counts.items():
        print(f"Patient ID: {patient_id}, Total Images: {sum(counts.values())}, Total Images Distribution: {sum(counts.values())/total_images*100:.2f}%")


    print("\nTotal class percentages:")
    for class_label, percentage in class_percentages.items():
        print(f"Class {class_label}: {percentage:.2f}%")

    print("\nArea statistics per class:")
    for class_label in sorted(df['c'].unique()):
        class_df = df[df['c'] == class_label]
        if not class_df.empty:
            mean_area = class_df['area'].mean()
            median_area = class_df['area'].median()
            std_area = class_df['area'].std() 
            # Assuming 'area' is a raw pixel count. Convert to percentage of a 512x512 image.
            # Total pixels in a 512x512 image = 512 * 512 = 262144
            total_image_pixels = 512 * 512
            mean_area_percent = (mean_area / total_image_pixels) * 100
            median_area_percent = (median_area / total_image_pixels) * 100
            std_area_percent = (std_area / total_image_pixels) * 100
            print(f"Class {class_label}: Mean Area = {mean_area_percent:.2f}%, Median Area = {median_area_percent:.2f}%, Std Dev Area = {std_area_percent:.2f}%")

if __name__ == "__main__":
    df = pd.read_excel("data/esaso_eval/cyst.xlsx")
    plot_patient_class_distribution(df)
    plot_total_class_distribution(df)
    print_analysis(df)