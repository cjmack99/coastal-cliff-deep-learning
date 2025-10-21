#!/usr/bin/env python3
"""
FAST Cliff Slope Extraction: Single DEM → Sample at Polygon Centers

Strategy:
1. Create ONE DEM for entire scene (not per polygon!)
2. For each polygon, sample the DEM along vertical transects
3. Compute slopes from transects

This is 100x faster than creating 22,850 individual DEMs!
"""

import numpy as np
import pandas as pd
import laspy
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import re
import platform
import os
from multiprocessing import Pool, cpu_count
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter
import time
import traceback

warnings.filterwarnings('ignore', category=UserWarning, module='pyproj')
warnings.filterwarnings('ignore', category=FutureWarning, module='pyproj')

# Configuration
LOCATIONS = [
    ('DelMar', 'DelMarPolygons595to620at10cm', '595', '620'),
    ('Encinitas', 'EncinitasPolygones708to764at10cm', '708', '764'),
    ('SanElijo', 'SanElijoPolygones684to708at10cm', '684', '708'),
    ('Solana', 'SolanaPolygones637to666at10cm', '637', '666'),
    ('Torrey', 'TorreyPolygones568to581at10cm', '568', '581'),
    ('Blacks', 'BlacksPolygones520to567at10cm', '520', '567')
]

HEIGHTS = {
    'DelMar': 30,
    'SanElijo': 40,
    'Solana': 50,
    'Encinitas': 50,
    'Torrey': 75,
    'Blacks': 100
}

BASE = ("/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
        if platform.system() == "Darwin"
        else "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs")

BASE_SHAPE_PATH = Path(os.path.join(BASE, "utilities", "shape_files"))
BASE_LAS_PATH = Path(os.path.join(BASE, "results"))
OUTPUT_BASE_PATH = Path(os.path.join(BASE, "results", "data_cubes", "slopes_optimized"))

# Processing parameters
DEM_RESOLUTION = 0.5         # Coarser is faster (0.5m is fine for cliffs)
ELEVATION_BIN_SIZE = 0.1     # 10 cm vertical bins
SMOOTH_SIGMA = 1.0           # More smoothing for stability
SLOPE_WINDOW = 5             # Window size for slope calculation

N_CORES = cpu_count()
N_PROCESSES = max(1, N_CORES // 4)


def create_scene_dem(points_x, points_y, points_z, bounds, resolution=DEM_RESOLUTION):
    """
    Create ONE DEM for the entire scene (not per polygon!).
    This is much faster than creating thousands of individual DEMs.
    
    Args:
        points_x, points_y, points_z: Point coordinates
        bounds: (xmin, ymin, xmax, ymax)
        resolution: Grid cell size
    
    Returns:
        dem: 2D elevation array
        x_coords: 1D array of X coordinates
        y_coords: 1D array of Y coordinates
        interpolator: Fast 2D interpolator for querying
    """
    xmin, ymin, xmax, ymax = bounds
    
    # Create regular grid
    x_coords = np.arange(xmin, xmax + resolution, resolution)
    y_coords = np.arange(ymin, ymax + resolution, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    print(f"    Creating DEM: {len(x_coords)} x {len(y_coords)} = {len(x_coords)*len(y_coords):,} cells")
    
    # Interpolate points to grid
    points = np.column_stack([points_x, points_y])
    
    # Use griddata with linear interpolation
    dem = griddata(points, points_z, (grid_x, grid_y), method='linear')
    
    # Smooth to reduce noise
    if SMOOTH_SIGMA > 0:
        valid_mask = ~np.isnan(dem)
        if valid_mask.any():
            dem_smooth = gaussian_filter(np.nan_to_num(dem, nan=0.0), sigma=SMOOTH_SIGMA)
            dem = np.where(valid_mask, dem_smooth, np.nan)
    
    # Create fast interpolator for querying
    # Replace NaNs with a fill value for interpolator
    dem_filled = np.nan_to_num(dem, nan=-9999)
    interpolator = RectBivariateSpline(y_coords, x_coords, dem_filled, kx=1, ky=1)
    
    return dem, x_coords, y_coords, interpolator


def sample_polygon_transect(polygon_geom, dem, x_coords, y_coords, max_height):
    """
    Sample the DEM along a vertical transect within a polygon.
    
    Instead of creating a new DEM, we just sample the existing one!
    
    Args:
        polygon_geom: Shapely polygon
        dem: Pre-computed DEM for entire scene
        x_coords, y_coords: DEM coordinate arrays
        max_height: Maximum elevation to consider
    
    Returns:
        DataFrame with z_bin, mean_x, n_cells for this polygon
    """
    # Get polygon bounds
    xmin, ymin, xmax, ymax = polygon_geom.bounds
    
    # Find DEM cells within polygon bounds
    x_mask = (x_coords >= xmin) & (x_coords <= xmax)
    y_mask = (y_coords >= ymin) & (y_coords <= ymax)
    
    if not x_mask.any() or not y_mask.any():
        return None
    
    # Get DEM subset
    x_indices = np.where(x_mask)[0]
    y_indices = np.where(y_mask)[0]
    
    # Extract sub-DEM
    y_start, y_end = y_indices[0], y_indices[-1] + 1
    x_start, x_end = x_indices[0], x_indices[-1] + 1
    
    sub_dem = dem[y_start:y_end, x_start:x_end]
    sub_x = x_coords[x_start:x_end]
    sub_y = y_coords[y_start:y_end]
    
    # Create meshgrid for sub-DEM
    sub_grid_x, sub_grid_y = np.meshgrid(sub_x, sub_y)
    
    # Mask to only cells with valid elevations
    valid_mask = ~np.isnan(sub_dem)
    
    if not valid_mask.any():
        return None
    
    # Get valid elevations and X positions
    valid_z = sub_dem[valid_mask]
    valid_x = sub_grid_x[valid_mask]
    
    # Bin by elevation
    elevation_bins = np.arange(ELEVATION_BIN_SIZE/2, max_height, ELEVATION_BIN_SIZE)
    
    results = []
    for z_center in elevation_bins:
        half_bin = ELEVATION_BIN_SIZE / 2
        mask = (valid_z >= z_center - half_bin) & (valid_z < z_center + half_bin)
        
        if not mask.any():
            continue
        
        x_values = valid_x[mask]
        n_cells = len(x_values)
        
        if n_cells >= 3:
            results.append({
                'z_bin': z_center,
                'mean_x': np.mean(x_values),
                'std_x': np.std(x_values),
                'n_cells': n_cells
            })
    
    if results:
        return pd.DataFrame(results)
    return None


def compute_slopes_from_transect(transect_df, window_size=SLOPE_WINDOW):
    """Compute local slopes from vertical transect."""
    if transect_df is None or len(transect_df) < 2:
        return None
    
    z_bins = transect_df['z_bin'].values
    x_values = transect_df['mean_x'].values
    
    slopes = np.full(len(z_bins), np.nan)
    half_window = window_size // 2
    
    for i in range(len(z_bins)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(z_bins), i + half_window + 1)
        
        if end_idx - start_idx < 3:
            continue
        
        z_window = z_bins[start_idx:end_idx]
        x_window = x_values[start_idx:end_idx]
        
        if len(z_window) >= 2:
            coeffs = np.polyfit(z_window, x_window, 1)
            dx_dz = coeffs[0]
            
            if abs(dx_dz) > 1e-6:
                dz_dx = 1.0 / dx_dz
                slopes[i] = abs(dz_dx)
    
    transect_df = transect_df.copy()
    transect_df['slope_magnitude'] = slopes
    transect_df['slope_angle_deg'] = np.degrees(np.arctan(slopes))
    
    return transect_df


def process_las_file(pathin, pathout_base, polys, location_name, overwrite=False):
    """
    Process LAS file by creating ONE DEM then sampling for all polygons.
    MUCH faster than creating 22,850 individual DEMs!
    """
    pathout_detail = Path(str(pathout_base) + '_slopes.csv')
    pathout_grid = Path(str(pathout_base) + '_slopes_grid.csv')
    pathout_summary = Path(str(pathout_base) + '_summary.csv')
    
    if pathout_detail.exists() and pathout_grid.exists() and not overwrite:
        return "SKIPPED"
    
    print(f"\n--- Processing: {pathin.name} ---")
    
    try:
        max_height = HEIGHTS.get(location_name, 30)
        print(f"  Max height: {max_height}m")
        
        # Read LAS
        print(f"  Reading LAS file...")
        with laspy.open(pathin) as lasf:
            las = lasf.read()
        
        points = las.points
        x_coords = np.array(points.x)
        y_coords = np.array(points.y)
        z_coords = np.array(points.z)
        
        print(f"  Loaded {len(x_coords):,} points")
        
        # Load polygons
        print(f"  Loading polygons...")
        polys_gdf = gpd.read_file(polys)
        polys_gdf["Polygon_ID"] = polys_gdf.index
        print(f"  Loaded {len(polys_gdf)} polygons")
        
        # Get overall bounds (union of all polygons)
        bounds = polys_gdf.total_bounds  # xmin, ymin, xmax, ymax
        
        # Filter points to polygon bounds (speeds up DEM creation)
        in_bounds = (
            (x_coords >= bounds[0]) & (x_coords <= bounds[2]) &
            (y_coords >= bounds[1]) & (y_coords <= bounds[3])
        )
        
        x_filtered = x_coords[in_bounds]
        y_filtered = y_coords[in_bounds]
        z_filtered = z_coords[in_bounds]
        
        print(f"  Points in bounds: {len(x_filtered):,} ({100*len(x_filtered)/len(x_coords):.1f}%)")
        
        if len(x_filtered) < 100:
            print("  ❌ Too few points in bounds")
            return None
        
        # CREATE ONE DEM FOR ENTIRE SCENE
        print(f"  Creating scene DEM (resolution={DEM_RESOLUTION}m)...")
        dem, dem_x, dem_y, interpolator = create_scene_dem(
            x_filtered, y_filtered, z_filtered, bounds
        )
        
        print(f"  DEM created: {dem.shape}")
        
        # Now process each polygon by SAMPLING the DEM (fast!)
        print(f"  Sampling DEM for {len(polys_gdf)} polygons...")
        all_results = []
        
        for poly_id in tqdm(range(len(polys_gdf)), desc="  Polygons", leave=False):
            poly_geom = polys_gdf.iloc[poly_id].geometry
            
            # Sample the DEM within this polygon
            transect_df = sample_polygon_transect(
                poly_geom, dem, dem_x, dem_y, max_height
            )
            
            if transect_df is None:
                continue
            
            # Compute slopes
            transect_df = compute_slopes_from_transect(transect_df)
            
            if transect_df is None:
                continue
            
            transect_df['polygon_id'] = poly_id
            all_results.append(transect_df)
        
        print(f"  Processed: {len(all_results)} / {len(polys_gdf)} polygons")
        
        if not all_results:
            print("  ❌ NO VALID RESULTS")
            return None
        
        # Save results
        df_all = pd.concat(all_results, ignore_index=True)
        
        pathout_detail.parent.mkdir(parents=True, exist_ok=True)
        
        df_all.to_csv(pathout_detail, index=False)
        
        slope_grid = df_all.pivot(
            index='polygon_id',
            columns='z_bin',
            values='slope_angle_deg'
        )
        slope_grid.to_csv(pathout_grid)
        
        summary = df_all.groupby('polygon_id').agg({
            'slope_angle_deg': ['mean', 'median', 'std', 'max'],
            'n_cells': 'sum'
        }).reset_index()
        summary.columns = ['polygon_id', 'mean_slope', 'median_slope', 'std_slope', 'max_slope', 'total_cells']
        summary.to_csv(pathout_summary, index=False)
        
        print(f"  ✓ Saved {len(df_all)} records from {len(all_results)} polygons")
        print(f"  ✓ Grid shape: {slope_grid.shape}")
        print(f"  ✓ Mean slope: {df_all['slope_angle_deg'].mean():.1f}°")
        
        return df_all
        
    except Exception as e:
        print(f"  ❌ EXCEPTION: {str(e)}")
        traceback.print_exc()
        return None


def process_single_file(args):
    """Wrapper for multiprocessing."""
    las_file, shp_path, location_name, polygon_name, overwrite = args
    
    try:
        date_match = re.match(r'^\d{8}', las_file.stem)
        date_str = date_match.group() if date_match else 'unknown'
        
        output_folder = OUTPUT_BASE_PATH / location_name / polygon_name
        pathout_base = output_folder / f"{date_str}_{polygon_name}"
        
        result = process_las_file(las_file, pathout_base, shp_path, location_name, overwrite)
        
        if result == "SKIPPED":
            return (True, las_file.name, "Skipped")
        elif result is not None:
            return (True, las_file.name, f"OK")
        else:
            return (False, las_file.name, "Failed")
        
    except Exception as e:
        return (False, las_file.name, f"Error: {str(e)}")


def process_location(location_name, polygon_name, start_mop, end_mop, overwrite=False):
    """Process all LAS files for a location."""
    max_height = HEIGHTS.get(location_name, 30)
    
    print(f"\n{'='*70}")
    print(f"Processing: {location_name}")
    print(f"  Max height: {max_height}m")
    print(f"  DEM resolution: {DEM_RESOLUTION}m")
    print(f"  Strategy: ONE DEM per file, sample for all polygons")
    print(f"  Using {N_PROCESSES} processes")
    print(f"{'='*70}")
    
    shp_path = BASE_SHAPE_PATH / polygon_name / f"{polygon_name}.shp"
    las_folder = BASE_LAS_PATH / location_name / 'noveg'
    
    if not shp_path.exists() or not las_folder.exists():
        print(f"  ⚠️  Path not found")
        return
    
    las_files = sorted(las_folder.glob('*noveg.las'))
    
    if not las_files:
        print(f"  ⚠️  No LAS files")
        return
    
    print(f"  Found {len(las_files)} LAS files\n")
    
    args_list = [(las_file, shp_path, location_name, polygon_name, overwrite)
                 for las_file in las_files]
    
    start_time = time.time()
    
    with Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc=f"  {location_name}"
        ))
    
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r[0])
    
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/len(las_files):.1f}s per file)")
    print(f"  Success: {successful}/{len(las_files)} files")


def main():
    """Main processing function."""
    print("\n" + "="*70)
    print("FAST CLIFF SLOPE EXTRACTION")
    print("Strategy: Single DEM per scene → Sample for all polygons")
    print("="*70)
    print(f"DEM Resolution: {DEM_RESOLUTION}m (coarser = faster)")
    print(f"Elevation Bins: {ELEVATION_BIN_SIZE}m")
    print(f"CPU Cores: {N_CORES}, Using: {N_PROCESSES} processes")
    print("\nLocation heights:")
    for loc, height in sorted(HEIGHTS.items()):
        print(f"  {loc}: {height}m")
    print("="*70)
    
    overall_start = time.time()
    
    for location_name, polygon_name, start_mop, end_mop in LOCATIONS:
        process_location(location_name, polygon_name, start_mop, end_mop, overwrite=False)
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*70)
    print(f"ALL COMPLETE! Total: {total_time/60:.1f} minutes")
    print("="*70)


if __name__ == "__main__":
    main()