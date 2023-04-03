# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

train_df = pd.read_csv('/Users/itimm/Downloads/asl-signs/train.csv')
print(f'Training data shape is: {train_df.shape}')
train_df.head()

print(f"Number of Unique Participants:                  {train_df['participant_id'].nunique()}")
print(f"Average Number of Rows Per Participant:         {train_df.groupby('participant_id').size().mean():.2f}")
print(f"Standard Deviation in Counts Per Participant:   {train_df.groupby('participant_id').size().std():.2f}")
print(f"Minimum Number of Examples For One Participant: {train_df.groupby('participant_id').size().min()}")
print(f"Maximum Number of Examples For One Participant  {train_df.groupby('participant_id').size().max()}")

train_df['sequence_id'].astype(str).describe()

print(f"Number of unique label: {train_df['sign'].nunique()}")

label_counts = train_df.sign.value_counts().to_frame().reset_index()
label_counts.columns = ['label','count']
plt.figure(figsize=(8,38))
#plt.barh('label', 'count', data=label_counts, height=0.5)
sns.barplot(y=label_counts['label'], x=label_counts['count'])
plt.show()

sample = pd.read_parquet('/Users/itimm/Downloads/asl-signs/train_landmark_files/16069/100015657.parquet')
print(sample.shape)
sample.head(10)

sample.describe()

print(f"There are {sample.frame.max()-sample.frame.min()} frames in this sequence.")

sample.type.value_counts()

#hand
edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(0,17),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20)]

def plot_frame(df, frame_id, ax):
    df = df[df.frame == frame_id].sort_values(['landmark_index'])
    x = list(df.x)
    y = list(df.y)
    
    ax.scatter(df.x, df.y, color='dodgerblue')
    for i in range(len(x)):
        ax.text(x[i], y[i], str(i))
        
    for edge in edges:
        ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color='salmon')
        ax.set_xlabel(f"Frame {frame_id}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    
def plot_frame_seq(df, frame_range, n_frames):
    frames = np.linspace(frame_range[0],frame_range[1],n_frames, dtype = int, endpoint=True)
    fig, ax = plt.subplots(n_frames, 1, figsize=(6,30))
    for i in range(n_frames):
        plot_frame(df, frames[i], ax[i])
        
    plt.show()
    
print("Left hand")
plot_frame_seq(sample[sample.type=='left_hand'], (178,186), 5)

sample[sample.type=='right_hand'].x.sum()

#Pose
def add_init_c(start, end, hand):
    return (
        pd.concat([hand['x'][start:start+1], hand['x'][end[0]:end[1]]]), 
        pd.concat([hand['y'][start:start+1], hand['y'][end[0]:end[1]]]), 
        pd.concat([hand['z'][start:start+1], hand['z'][end[0]:end[1]]])
        )

def plot_pose(pose, td):
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    if td:
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

    ind = [[0, [1, 4]], [3, [7, 8]], [0, [4, 7]], [6, [8, 9]], [9, [10, 11]], [11, [12, 13]], [12, [14, 15]], [14, [16, 17]], 
           [16, [22, 23]], [16, [18, 19]], [16, [20, 21]], [18, [20, 21]], [11, [13, 14]], [13, [15, 16]], [15, [21, 22]], [15, [19, 20]],
           [15, [17, 18]], [17, [19, 20]], [12, [24, 25]], [24, [26, 27]], [26, [28, 29]], [28, [30, 31]], [30, [32, 33]], [28, [32, 33]], 
           [11, [23, 24]], [23, [25, 26]], [25, [27, 28]], [27, [29, 30]], [29, [31, 32]], [27, [31, 32]], [23, [24, 25]]]

    for i, k in ind: 
        x, y, z = add_init_c(i, k, pose)
        if td:
            ax.plot(x, -1*y, z)
        else:
            ax.plot(x, -1*y)

plot_pose(sample.loc[sample.type=='pose'][:33], True)

#Pose 5 frames
plot_pose(sample.loc[sample.type=='pose'][:33], False)
plot_pose(sample.loc[sample.type=='pose'][33: 66], False)
plot_pose(sample.loc[sample.type=='pose'][66: 99], False)
plot_pose(sample.loc[sample.type=='pose'][99: 132], False)
plot_pose(sample.loc[sample.type=='pose'][132: 165], False)

print("First frame in this sequence:")
plot_pose(sample.loc[sample.type=='pose'][: 33], False)

print("Middle frame in this sequence:")
plot_pose(sample.loc[sample.type=='pose'][33*52: 33*53], False)

print("Last frame in this sequence:")
plot_pose(sample.loc[sample.type=='pose'][33*104: 33*106], False)

#Face
def plot_face(face, td):
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    if td:
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

    if td:
        ax.scatter(face['x'], -1*face['y'], face['z'])
    else:
        ax.scatter(face['x'], -1*face['y'])
        
plot_face(sample.loc[sample.type=='face'][: 468], True)

plot_face(sample.loc[sample.type=='face'][: 468], False)
plot_face(sample.loc[sample.type=='face'][468: 468*2], False)
plot_face(sample.loc[sample.type=='face'][468*2: 468*3], False)
plot_face(sample.loc[sample.type=='face'][468*3: 468*4], False)
plot_face(sample.loc[sample.type=='face'][468*4: 468*5], False)

print("First frame in this sequence:")
plot_face(sample.loc[sample.type=='face'][: 468], False)

print("Middle frame in this sequence:")
plot_face(sample.loc[sample.type=='face'][468*52: 468*53], False)

print("Last frame in this sequence:")
plot_face(sample.loc[sample.type=='face'][468*104: 468*106], False)

def get_details_per_sign(sign):
    train_sign_sample = train_df[train_df['sign'] == sign]
    n_frames = 0
    n_left_hand = 0
    n_right_hand = 0
    n_face = 0
    n_both_hands = 0
    for _,row in train_sign_sample.iterrows():
        df = pd.read_parquet(os.path.join("/kaggle/input/asl-signs", row.path))
        n_frames += df['frame'].nunique()
        n_left_hand += np.sum(df[(df['type'] == 'left_hand') & (df['landmark_index'] == 0)]['x'].isnull() == False)
        n_right_hand += np.sum(df[(df['type'] == 'right_hand') & (df['landmark_index'] == 0)]['x'].isnull() == False)
        n_face += np.sum(df[(df['type'] == 'face') & (df['landmark_index'] == 0)]['x'].isnull() == False)
        
        df_both_hands = df[(df['type'] == 'left_hand') & (df['landmark_index'] == 0)].merge(\
                            df[(df['type'] == 'right_hand') & (df['landmark_index'] == 0)], on='frame', suffixes=('_left', '_right'))
        n_both_hands += df_both_hands[(df_both_hands['x_left'].isnull() == False) &\
                                             (df_both_hands['x_right'].isnull() == False)]['frame'].count()
            
    return n_frames/len(train_sign_sample), n_left_hand/n_frames, n_right_hand/n_frames, n_both_hands/n_frames, n_face/n_frames

for sign in ['cloud', 'thankyou', 'donkey', 'because', 'yellow', 'icecream']:
    total_frames, pct_left, pct_right, pct_both, pct_face = get_details_per_sign(sign)
    print("="*20, f"{sign}", "="*20)
    print(f"Average Number of Frames per Sequence = {total_frames}")
    print(f"Percent of Frames in which a body part exists: Left Hand: {pct_left*100:.02f} %, Right Hand: {pct_right*100:.02f} %, Both Hands: {pct_both*100:.02f} %, Face: {pct_face*100:.02f} %")
    print()
    
    

