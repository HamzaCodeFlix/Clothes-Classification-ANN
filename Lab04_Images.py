# import cv2
# import matplotlib.pyplot as plt
# image1=cv2.imread('dwn.jpg')
# cv2.imshow('image',image1)

# print("shape of image:",image1.shape)
# print("type of image:",type(image1))
# cv2.waitKey(0)
# plt.figure()
# plt.imshow(image1)
# plt.title('Image')
# plt.show()
# #############################################

#  # best resoltion is 2024 x 2024 
# image2=cv2.resize(image1,(500,500))
# image3=cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
# image4=cv2.cvtColor(image3,cv2.COLOR_RGB2GRAY)

# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(image1)
# plt.title('Original Image')

# plt.subplot(2,2,2)
# plt.imshow(image2)
# plt.title('Resized Image')

# plt.subplot(2,2,3)
# plt.imshow(image3)
# plt.title(' RGB to BGR Image')

# plt.subplot(2,2,4)
# plt.imshow(image4,cmap='gray')
# plt.title(' RGB to Gray Image')

# plt.show()


# ######################################################

# import csv
# import os
# labels={'Coat':0,'Jeans':1,'Polo':3,'Shirt':4,'Sweater':5,'T-shirts':6}
# l=list(labels.keys())
# with open("Myclothesproject.csv","w",newline='') as fl:
#     for i in l:   # shirts, coat 
#         wr=csv.writer(fl)
#         path='C:\\ Users\\ Muhammad Hamza\\ ANNDL_Fall_2026\\ ANN-Clothes-Project\\ Clothes_Dataset\\ {}'.format(i)
#         for r,d,f in os.walk(path):
#             for fname in f:
#                 if '.jpg' or '.png' in fname:
#                     record=[]
#                     file_path=os.path.join(path,fname)
#                     image7=cv2.imread(file_path)
#                     image7=cv2.resize(image7,(530,200))
#                     image7=image7.flatten()
#                     label1=labels[i]
#                     record.append(label1)
#                     record.extend(image7)
#                     wr.writerow(record)
                    










################################################


import cv2
import csv
import os

# -----------------------------
# 1. Labels
# -----------------------------
labels = {
    'Coat': 0,
    'Jeans': 1,
    'Polo': 2,
    'Shirt': 3,
    'Sweater': 4,
    'T-shirts': 5
}

# -----------------------------
# 2. Dataset path
# -----------------------------
dataset_path = r"C:\Users\Muhammad Hamza\ANNDL_Fall_2026\ANN-Clothes-Project\Clothes_Dataset"

# -----------------------------
# 3. CSV output file
# -----------------------------
csv_file = "Myclothesproject.csv"

# -----------------------------
# 4. Create CSV
# -----------------------------
with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

    # Process each clothing category
    for category, label in labels.items():

        folder_path = os.path.join(dataset_path, category)

        print(f"Processing: {category}")

        # Check folder exists
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # Go through all files
        for filename in os.listdir(folder_path):

            # Only process image files
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):

                file_path = os.path.join(folder_path, filename)

                # Read image
                image = cv2.imread(file_path)

                # Skip corrupted/unreadable images
                if image is None:
                    print(f"Could not read: {file_path}")
                    continue

                # Resize image
                image = cv2.resize(image, (530, 200))

                # Convert image into 1D array
                image = image.flatten()

                # Label + image pixels
                record = [label]
                record.extend(image)

                # Write row
                writer.writerow(record)

print("CSV file created successfully!")
print(f"Saved as: {csv_file}")