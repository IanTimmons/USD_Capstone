import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import csv


root = "data\\train_landmark_files"
participant_id = []
file_count = {}


for path, dirs, files in os.walk(root):
    for subdir in dirs:
        participant_id.append(subdir)

for id in participant_id:
    temp_path = root + "\\" + id
    count = 0
    for path in os.listdir(temp_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(temp_path, path)):
            count += 1
    #print("Participant ID:",id,'- File Count:', count)
    file_count[id]=count    

# Readable text file
df = open('EDA_data.txt', 'w+')
df.write("File Count by Participant ID: ")
df.write('\n')

for key, value in file_count.items():
    lines = str("Participant Id:" + str(key) + " File Count:" + str(value))
    df.write(lines)
    df.write('\n')
df.close()

# Write data to CSV 
with open('file_counts.csv', 'w+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Participant ID", "File Count"])
    for key, value in file_count.items():
        writer.writerow([key,value])


# File Count By Participant ID
keys = list(file_count.keys())
vals = [float(file_count[k]) for k in keys]

sns.barplot(x=keys, y=vals)
plt.xticks(rotation=45)
plt.xlabel('Participant ID')
plt.ylabel('File Count')
plt.title('File Count by Participant ID')
plt.show()
plt.savefig('File Count by Participant ID')

