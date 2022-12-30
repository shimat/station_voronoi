import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import get_station_locations

IMAGE_SIZE = 1000

station_locations = get_station_locations("geojson/函館本線.csv")
print(station_locations)

station_locations = MinMaxScaler((0, IMAGE_SIZE)).fit_transform(station_locations)#.astype(int)
print(station_locations)

#min_lat, max_lat, _, _ = cv2.minMaxLoc(station_locations[:, 0])
#print(min_lat, max_lat)
#min_lon, max_lon, _, _ = cv2.minMaxLoc(station_locations[:, 1])
#print(min_lon, max_lon)

subdiv = cv2.Subdiv2D((0, 0, IMAGE_SIZE+1, IMAGE_SIZE+1))

for p in station_locations:
    subdiv.insert((p[0], p[1]))

facets, centers = subdiv.getVoronoiFacetList([])
#print(facets)

img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
for p in centers:
    cv2.drawMarker(img, (int(p[0]), int(p[1])), (0, 0, 255), thickness=2)
cv2.polylines(img, [f.astype(int) for f in facets], True, (255, 255, 255), thickness=1)
img = cv2.flip(img, 0)

cv2.imwrite('voronoi.png', img)

"""
fig, ax = plt.subplots()
ax.scatter(station_locations[:, 1], station_locations[:, 0], c='blue', s=10, label='original')
#ax.scatter(station_locations_n[:, 1], station_locations_n[:, 0], c='red', s=10, label='normalized')
ax.grid()
plt.savefig('fig.png')
"""