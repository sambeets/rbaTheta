import pysal as ps
import libpysal
from libpysal.weights import Queen, Rook, KNN, Kernel, DistanceBand
import geopandas
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, scatter, show


fig = figure(figsize=(9,9))
shp_path = r'docs\QGISfiles\wind_park_thiessen.shp'

df = geopandas.read_file(shp_path)
qW = Queen.from_dataframe(df)
centroids = np.array([[poly.centroid.x, poly.centroid.y] for poly in df.geometry])
plt.plot(centroids[:, 0], centroids[:, 1], '.')

#plt.plot(s04[:,0], s04[:,1], '-')
#plt.ylim([59.37,59.385])
for k, neighbours in qW.neighbors.items():
    origin = centroids[k]
    for neighbour in neighbours:
        segment = centroids[[k, neighbour]]
        plt.plot(segment[:, 0], segment[:, 1], '-')
plt.title('Queen Neighbor Graph')
#fig.savefig(r"plotted_figures\foo.pdf", bbox_inches='tight')
plt.show()


#-----------------------
fig = figure(figsize=(9,9))
shp_path = r'docs\QGISfiles\wind_park_thiessen.shp'


df = geopandas.read_file(shp_path)
points = [(poly.centroid.y, poly.centroid.x) for poly in df.geometry]
radius_km = libpysal.cg.sphere.RADIUS_EARTH_KM
threshold = libpysal.weights.min_threshold_dist_from_shapefile(shp_path, radius=radius_km) # Maximum nearest neighbor distance between the n observations
print(threshold*.025)
print(threshold)
print(points)
w = DistanceBand(points, threshold=threshold*.025, binary=False)
print(w)
centroids = np.array([[poly.centroid.x, poly.centroid.y] for poly in df.geometry])

plt.plot(centroids[:, 0], centroids[:, 1], '.')
#plt.plot(s04[:,0], s04[:,1], '-')
#plt.ylim([59.37,59.385])
for k, neighbours in w.neighbors.items():
    origin = centroids[k]
    for neighbour in neighbours:
        segment = centroids[[k, neighbour]]
        plt.plot(segment[:, 0], segment[:, 1], '-')
plt.title('Distance band Neighbor Graph')
#fig.savefig("plotted_figures\foo.pdf", bbox_inches='tight')
plt.show()



