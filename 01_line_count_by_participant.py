import os
import pyarrow.parquet as pq
import csv

root = "data\\train_landmark_files"
participant_id = []
line_count_dict = {}


for path, dirs, files in os.walk(root):
    for subdir in dirs:
        participant_id.append(subdir)
print(participant_id)

for id in participant_id:
    temp_path = root + "\\" + id
    count = 0
    for path in os.listdir(temp_path):
        table_path = temp_path + "\\" + path
        table = pq.read_table(table_path, columns=[])
        count += table.num_rows
    line_count_dict[id] = count
    print("Participant ID:",id,'- Line Count:', count)

# Readable text file
df = open('Participant_line_data.txt', 'w+')
df.write("Line Count by Participant ID: ")
df.write('\n')

for key, value in line_count_dict.items():
    lines = str("Participant Id:" + str(key) + " Line Count:" + str(value))
    df.write(lines)
    df.write('\n')
df.close()

# Write data to CSV 
with open('Participant_line_data.csv', 'w+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Participant ID", "Line Count"])
    for key, value in line_count_dict.items():
        writer.writerow([key,value])