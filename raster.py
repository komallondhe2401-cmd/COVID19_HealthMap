import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import xarray as xr
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestRegressor

# 1) Load district polygons with case counts and population
districts = gpd.read_file("india_districts.geojson").to_crs("EPSG:4326")
districts["cases_per_100k"] = districts["cases"] / districts["population"] * 100_000

# 2) Create weighted points for KDE (using centroids)
pts = districts.copy()
pts["geometry"] = pts.geometry.centroid
X = np.vstack([pts.geometry.x.values, pts.geometry.y.values]).T
weights = pts["cases_per_100k"].values

# 3) Fit KDE and render to raster grid
kde = KernelDensity(bandwidth=0.5, kernel="gaussian")  # tune bandwidth
kde.fit(X, sample_weight=weights)

# Define grid over India
lon = np.linspace(68, 97, 600)
lat = np.linspace(7.5, 37.5, 600)
Lon, Lat = np.meshgrid(lon, lat)
grid = np.vstack([Lon.ravel(), Lat.ravel()]).T
log_density = kde.score_samples(grid)
density = np.exp(log_density).reshape(Lon.shape)

# 4) Save KDE surface as GeoTIFF
transform = rasterio.transform.from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), Lon.shape[1], Lat.shape[0])
with rasterio.open(
    "covid_kde.tif", "w",
    driver="GTiff", height=Lat.shape[0], width=Lon.shape[1],
    count=1, dtype="float32", crs="EPSG:4326", transform=transform
) as dst:
    dst.write(density.astype(np.float32), 1)

# 5) Dasymetric redistribution (conceptual snippet)
# Load population raster
pop = rasterio.open("worldpop_india.tif")
# For each district, clip pop, compute weights, and write redistributed cases raster
# (Loop omitted for brevity—use rasterio.mask and per-district normalization)

# 6) ML with raster covariates (tabular extraction)
# Suppose we have features as rasters aligned to a common grid
features = {
    "pop_density": xr.open_rasterio("worldpop_india_resampled.tif")[0].values,
    "pm25": xr.open_rasterio("pm25_resampled.tif")[0].values,
    "ntl": xr.open_rasterio("viirs_resampled.tif")[0].values
}
target = xr.open_rasterio("cases_dasymetric.tif")[0].values

# Build dataset
mask = ~np.isnan(target)
X_tab = np.column_stack([f[mask] for f in features.values()])
y_tab = target[mask]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_tab, y_tab)
