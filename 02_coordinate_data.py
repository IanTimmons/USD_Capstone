import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = "data\\train_landmark_files\\2044\\3127189.parquet"

table = pd.read_parquet(root)
column_names = list(table.columns)
print("Column Names:")
print(column_names)
print("\n")
"""
frame_list = [table['frame'].unique()]
print("Frames:")
print(frame_list)
print("Length: ", len(frame_list))
print("\n")

row_id_list = [table['row_id'].unique()]
print("Row Id:")
print(row_id_list)
print("Length: ", len(table['row_id']))
print("\n")

# Unique Values for 'type' column
type_list = [table['type'].unique()]
print("Types:")
print(type_list)
print("\n")

landmark_list = [table['landmark_index'].unique()]
print("Landmark Index:")
print(landmark_list)
print("Length: ", len(table['landmark_index']))
print("\n")

#Min/Max Values
x_min = [table['x'].min()]
y_min = [table['y'].min()]
z_min = [table['z'].min()]

print("X-Min:", x_min)
print("Y-Min:", y_min)
print("Z-Min:", z_min)

x_max = [table['x'].max()]
y_max = [table['y'].max()]
z_max = [table['z'].max()]

print("X-Max:", x_max)
print("Y-Max:", y_max)
print("Z-Max:", z_max)
"""
sns.scatterplot(data=table, x='x', y='y', hue='type')
plt.xlabel('X-Coordinates')
plt.ylabel('Y-Coordinates')
plt.title('Plot of Parquet File 3127189: Sign for Eye')
plt.savefig('Participant 2044- Eye')
plt.show()
