import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "data\\train.csv"
df = pd.read_csv(path)
#print(df.head())

# Readable text file
text = open('03_training_csv_breakdown.txt', 'w+')
text.write("General Data Information: ")
text.write('\n')
text.write("Unique Words:")
text.write('\n')
list_unique = df['sign'].unique()
text.write(str(list_unique))
text.write("\n")
print(list_unique)
print(len(list_unique))
word_freq = df['sign'].value_counts()
text.write("Word Frequency: ")
text.write("\n")
text.write(str(word_freq))
text.write("\n")
text.write('\n')

#Word Count by Participant ID
text.write("Word Count by Participant ID: ")
text.write('\n')
text.write("Participant 2044:")
text.write('\n')
par_id_2044 = df[df['participant_id']==2044]
#print(par_id_2044.head())
word_freq_2044 = par_id_2044['sign'].value_counts()
print(word_freq_2044)
text.write(str(word_freq_2044))
text.write('\n')
text.write('\n')

text.write("Participant 4718:")
text.write('\n')
par_id_4718 = df[df['participant_id']==4718]
#print(par_id_2044.head())
word_freq_4718 = par_id_4718['sign'].value_counts()
print(word_freq_4718)
text.write(str(word_freq_4718))
text.write('\n')



