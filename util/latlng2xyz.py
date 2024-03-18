# import math

# def latlon_to_xyz(lat,lon):
#     """Convert angluar to cartesian coordiantes

#     latitude is the 90deg - zenith angle in range [-90;90]
#     lonitude is the azimuthal angle in range [-180;180] 
#     """
#     r = 6371 # https://en.wikipedia.org/wiki/Earth_radius
#     theta = math.pi/2 - math.radians(lat) 
#     phi = math.radians(lon)
#     x = r * math.sin(theta) * math.cos(phi) # bronstein (3.381a)
#     y = r * math.sin(theta) * math.sin(phi)
#     z = r * math.cos(theta)
#     return x, y, z


# Converting lat/long to cartesian
import numpy as np

def latlon_to_xyz(lat=None,lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z

def PlateCarree_to_xyz(lat, lon, min_lat, min_lon):
    return (lon - min_lon) * 1000, (lat - min_lat) * 1000, 0
