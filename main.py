import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import get_all_station_locations

IMAGE_SIZE = 1000


def show_plot(x, y, file_name="fig.png"):
    fig, ax = plt.subplots()
    ax.scatter(x, y, c='blue', s=10)
    ax.grid()
    st.pyplot(fig)



station_locations = get_all_station_locations("csv")
print(station_locations)

station_locations = MinMaxScaler((0, IMAGE_SIZE)).fit_transform(station_locations)#.astype(int)
#print(station_locations)
show_plot(station_locations[:, 0], station_locations[:, 1])

#min_lat, max_lat, _, _ = cv2.minMaxLoc(station_locations[:, 0])
#print(min_lat, max_lat)
#min_lon, max_lon, _, _ = cv2.minMaxLoc(station_locations[:, 1])
#print(min_lon, max_lon)

subdiv = cv2.Subdiv2D((0, 0, IMAGE_SIZE+1, IMAGE_SIZE+1))

for p in station_locations:
    subdiv.insert((p[0], p[1]))

facets, centers = subdiv.getVoronoiFacetList([])
#print(facets)

img = np.zeros((IMAGE_SIZE+100, IMAGE_SIZE+100, 3), np.uint8)
for p in centers:
    cv2.drawMarker(img, (p+50).astype(int), (0, 0, 255), thickness=2)
cv2.polylines(img, [(f+50).astype(int) for f in facets], True, (255, 255, 255), thickness=1)
img = cv2.flip(img, 0)

#cv2.imwrite('voronoi.png', img)
st.image(img, channels="BGR")