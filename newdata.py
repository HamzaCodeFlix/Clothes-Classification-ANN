import cv2
import csv
import os

labels = {
    'Coat': 0,
    'Jeans': 1,
    'Shirt': 2
}

l = list(labels.keys())

# Correct dataset path
base_path = r"C:\Users\Muhammad Hamza\ANNDL_Fall_2026\ANN-Clothes-Project\Clothes_Dataset"

with open("Images224.csv", "w", newline="") as fl:

    wr = csv.writer(fl)

    for i in l:

        # Create path for each category
        path = os.path.join(base_path, i)

        print("Processing:", path)

        for r, d, f in os.walk(path):

            for fname in f:

                # Check image files correctly
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):

                    record = []

                    # Full image path
                    file_path = os.path.join(r, fname)

                    # Read image
                    image7 = cv2.imread(file_path)

                    if image7 is None:
                        print("Could not read:", file_path)
                        continue

                    # Resize to 28 x 28
                    image7 = cv2.resize(image7, (224, 224))

                    # Flatten image
                    image7 = image7.flatten()

                    # Get label
                    label1 = labels[i]

                    # Add label
                    record.append(label1)

                    # Add image pixels
                    record.extend(image7)

                    # Write to CSV
                    wr.writerow(record)

print("Images28.csv created successfully!")