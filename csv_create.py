import os
import csv

folder_0 = "data/train/Car"  # car = 0
folder_1 = "data/train/Truck"  # truck = 1

csv_filename = "data.csv"

def get_labeled_files(folder_path, label):
    files = [f for f in os.listdir(folder_path) if f.endswith(".jpeg")]
    files.sort()
    return [(f, label) for f in files]

all_data = []
all_data += get_labeled_files(folder_0, 0)
all_data += get_labeled_files(folder_1, 1)

with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "label"])
    for filename, label in all_data:
        writer.writerow([filename, label])

print(f"CSV сохранён как {csv_filename}")
