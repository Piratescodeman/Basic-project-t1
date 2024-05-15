import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Function to get color name from RGB values
def get_color_name(R, G, B, color_data):
    minimum = 1000
    color_name = ''
    for i in range(len(color_data)):
        d = abs(R - color_data.loc[i, 'R']) + abs(G - color_data.loc[i, 'G']) + abs(B - color_data.loc[i, 'B'])
        if d < minimum:
            minimum = d
            color_name = color_data.loc[i, 'color_name']
    return color_name

# Function to display the dominant colors and their names
def display_colors(colors, color_data):
    img_colors = np.zeros((100, len(colors) * 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        img_colors[:, i*100:(i+1)*100] = color
        color_name = get_color_name(color[2], color[1], color[0], color_data)
        cv2.putText(img_colors, color_name, (i*100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('Dominant Colors', img_colors)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load image
img_path = 'pic2.jpg'
img = cv2.imread(img_path)

# Reshape image
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)

# K-means clustering
num_colors = 5
kmeans = KMeans(n_clusters=num_colors)
kmeans.fit(pixels)
colors = kmeans.cluster_centers_
colors = np.uint8(colors)

# Load color data from CSV
color_data = pd.read_csv('colors.csv')

# Display dominant colors
display_colors(colors, color_data)
