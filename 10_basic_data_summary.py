import pandas as pd
import csv


#Basic File Data by Participant
df = pd.read_csv("file_counts.csv", header=0)

file_count_min = df['File Count'].min()
print("File Count Minimum: ",file_count_min)

file_count_max = df['File Count'].max()
print("File Count Maximum: ",file_count_max)

file_count_avg = df['File Count'].mean()
print("File Count Average: ", round(file_count_avg))


#Basic Line Count Data by Participant
line_df = pd.read_csv("line_counts.csv", header=0)

line_count_min = line_df['Line Count'].min()
print("Line Count Minimum: ",line_count_min)

line_count_max = line_df['Line Count'].max()
print("Line Count Maximum: ",line_count_max)

line_count_avg = line_df['Line Count'].mean()
print("Line Count Average: ", round(line_count_avg))


#Looking at one participant
