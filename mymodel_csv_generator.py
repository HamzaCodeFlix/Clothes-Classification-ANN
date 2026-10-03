import cv2
import csv
import os

labels = {'Coat': 0,'Jeans': 1, 'Shirt': 2,'Sweater': 3 }

dataset_path = r"C:\Users\Muhammad Hamza\ANNDL_Fall_2026\ANN-Clothes-Project\Clothes_Dataset"

with open("Images28.csv", "w", newline="") as file:
    writer = csv.writer(file)

    for i in labels:
        path = os.path.join(dataset_path, i)
        
        for r, d, f in os.walk(path):

            for fname in f:

                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    record = []
                    file_path = os.path.join(path, fname)
                    image7 = cv2.imread(file_path)
                    image7 = cv2.resize(image7, (28, 28))
                    image7 = cv2.cvtColor(image7, cv2.COLOR_BGR2RGB)
                    image7 = image7.flatten()
                    label1 = labels[i]
                    record.append(label1)
                    record.extend(image7)
                    writer.writerow(record)

print("Images28.csv created successfully!")


