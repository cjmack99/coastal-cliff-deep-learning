#!/usr/bin/env python3
"""
Optimized Cliff Slope Extraction: Top-Down DEM → Vertical Transects → Slopes
WITH DEBUG OUTPUT
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
from scipy.interpolate import griddata, interp1d
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

# Location-specific maximum heights (in meters)
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
DEM_RESOLUTION = 0.25        # Top-down DEM resolution (m)
ELEVATION_BIN_SIZE = 0.1     # Vertical bins (10 cm)
MIN_POINTS_FOR_DEM = 50      # Minimum points needed per polygon
SMOOTH_SIGMA = 0.5           # Gaussian smoothing for DEM
SLOPE_WINDOW = 5             # Window size for slope calculation

N_CORES = cpu_count()
N_PROCESSES = max(1, N_CORES // 4)


def create_topdown_dem(points_df, polygon_geom, resolution=DEM_RESOLUTION):
    """Create traditional top-down DEM from points within polygon."""
    if len(points_df) < MIN_POINTS_FOR_DEM:
        return None, None, None, None
    
    # Get polygon bounds
    xmin, ymin, xmax, ymax = polygon_geom.bounds
    
    # Create regular grid
    x_coords = np.arange(xmin, xmax + resolution, resolution)
    y_coords = np.arange(ymin, ymax + resolution, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Interpolate points to grid
    points = points_df[['X', 'Y']].values
    values = points_df['Z'].values
    
    # Use linear interpolation
    dem = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # Optional: smooth to reduce noise
    if SMOOTH_SIGMA > 0:
        valid_mask = ~np.isnan(dem)
        if valid_mask.any():
            dem_smooth = gaussian_filter(np.nan_to_num(dem, nan=0.0), sigma=SMOOTH_SIGMA)
            dem = np.where(valid_mask, dem_smooth, np.nan)
    
    return dem, (xmin, xmax, ymin, ymax), x_coords, y_coords


def extract_vertical_transect(dem, x_coords, y_coords, elevation_bins):
    """Extract vertical transect from DEM."""
    if dem is None:
        return None
    
    results = []
    
    for z_center in elevation_bins:
        half_bin = ELEVATION_BIN_SIZE / 2
        mask = (dem >= z_center - half_bin) & (dem < z_center + half_bin)
        
        if not mask.any():
            continue
        
        row_indices, col_indices = np.where(mask)
        x_values = x_coords[col_indices]
        
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


def process_polygon(points_df, polygon_geom, polygon_id, max_height):
    """Complete processing pipeline for one polygon."""
    # Step 1: Create DEM
    dem, extent, x_coords, y_coords = create_topdown_dem(points_df, polygon_geom)
    
    if dem is None:
        return None
    
    # Step 2: Define elevation bins
    elevation_bins = np.arange(ELEVATION_BIN_SIZE/2, max_height, ELEVATION_BIN_SIZE)
    
    # Extract transect
    transect_df = extract_vertical_transect(dem, x_coords, y_coords, elevation_bins)
    
    if transect_df is None:
        return None
    
    # Step 3: Compute slopes
    transect_df = compute_slopes_from_transect(transect_df)
    
    if transect_df is None:
        return None
    
    # Add polygon ID
    transect_df['polygon_id'] = polygon_id
    
    return transect_df


def process_las_file(pathin, pathout_base, polys, location_name, overwrite=False):
    """Process entire LAS file for all polygons WITH DEBUG OUTPUT."""
    pathout_detail = Path(str(pathout_base) + '_slopes.csv')
    pathout_grid = Path(str(pathout_base) + '_slopes_grid.csv')
    pathout_summary = Path(str(pathout_base) + '_summary.csv')
    
    # Check if exists
    if pathout_detail.exists() and pathout_grid.exists() and not overwrite:
        # Return a success indicator so it doesn't count as failure
        return "SKIPPED"
    
    print(f"\n--- Processing: {pathin.name} ---")
    
    try:
        # Get max height for this location
        max_height = HEIGHTS.get(location_name, 30)
        print(f"  Max height: {max_height}m")
        
        # Read LAS
        print(f"  Reading LAS file...")
        with laspy.open(pathin) as lasf:
            las = lasf.read()
        
        print(f"  Loaded {len(las.x):,} points")
        print(f"  LAS bounds: X=[{las.x.min():.1f}, {las.x.max():.1f}], Y=[{las.y.min():.1f}, {las.y.max():.1f}], Z=[{las.z.min():.1f}, {las.z.max():.1f}]")
        
        # Load polygons FIRST to get CRS
        print(f"  Loading polygons from: {polys}")
        polys_gdf = gpd.read_file(polys)
        polys_gdf["Polygon_ID"] = polys_gdf.index
        print(f"  Loaded {len(polys_gdf)} polygons (CRS: {polys_gdf.crs})")
        print(f"  Polygon bounds: {polys_gdf.total_bounds}")
        
        # Create points dataframe using the polygon CRS
        df_pts = pd.DataFrame({
            'X': las.x,
            'Y': las.y,
            'Z': las.z
        })
        
        print(f"  Creating GeoDataFrame with CRS: {polys_gdf.crs}")
        gdf_pts = gpd.GeoDataFrame(
            df_pts,
            geometry=[Point(x, y) for x, y in zip(las.x, las.y)],
            crs=polys_gdf.crs
        )
        
        # Spatial join
        print(f"  Performing spatial join...")
        joined = gpd.sjoin(gdf_pts, polys_gdf[['Polygon_ID', 'geometry']],
                           how='inner', predicate='within')
        
        print(f"  Points in polygons: {len(joined):,} / {len(las.x):,} ({100*len(joined)/len(las.x):.1f}%)")
        
        if len(joined) == 0:
            print("  ❌ NO POINTS IN POLYGONS!")
            print("  This means:")
            print("    - CRS mismatch between LAS and shapefile")
            print("    - Or points are outside polygon boundaries")
            return None
        
        # Check points per polygon
        points_per_poly = joined.groupby('Polygon_ID').size()
        print(f"  Points per polygon: min={points_per_poly.min()}, mean={points_per_poly.mean():.0f}, max={points_per_poly.max()}")
        print(f"  Polygons with >= {MIN_POINTS_FOR_DEM} points: {(points_per_poly >= MIN_POINTS_FOR_DEM).sum()} / {len(polys_gdf)}")
        
        # Process each polygon
        all_results = []
        failed_polygons = 0
        
        for poly_id in tqdm(range(len(polys_gdf)), desc="  Polygons", leave=False):
            poly_points = joined[joined['Polygon_ID'] == poly_id]
            
            if len(poly_points) < MIN_POINTS_FOR_DEM:
                failed_polygons += 1
                continue
            
            poly_geom = polys_gdf.iloc[poly_id].geometry
            
            result = process_polygon(poly_points, poly_geom, poly_id, max_height)
            
            if result is not None:
                all_results.append(result)
            else:
                failed_polygons += 1
        
        print(f"  Processed: {len(all_results)} polygons succeeded, {failed_polygons} failed")
        
        if not all_results:
            print("  ❌ NO VALID RESULTS - all polygons failed processing")
            return None
        
        # Concatenate all results
        df_all = pd.concat(all_results, ignore_index=True)
        
        # Ensure output directory exists
        pathout_detail.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed format
        print(f"  Writing to: {pathout_detail}")
        df_all.to_csv(pathout_detail, index=False)
        
        # Create grid format
        slope_grid = df_all.pivot(
            index='polygon_id',
            columns='z_bin',
            values='slope_angle_deg'
        )
        slope_grid.to_csv(pathout_grid)
        
        # Create summary statistics
        summary = df_all.groupby('polygon_id').agg({
            'slope_angle_deg': ['mean', 'median', 'std', 'max'],
            'n_cells': 'sum'
        }).reset_index()
        summary.columns = ['polygon_id', 'mean_slope', 'median_slope', 'std_slope', 'max_slope', 'total_cells']
        summary.to_csv(pathout_summary, index=False)
        
        print(f"  ✓ Saved {len(df_all)} records from {len(all_results)} polygons")
        print(f"  ✓ Grid shape: {slope_grid.shape}")
        print(f"  ✓ Mean slope: {df_all['slope_angle_deg'].mean():.1f}°")
        print(f"  ✓ Files written:")
        print(f"     - {pathout_detail.name}")
        print(f"     - {pathout_grid.name}")
        print(f"     - {pathout_summary.name}")
        
        return df_all
        
    except Exception as e:
        print(f"  ❌ EXCEPTION: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()
        return None


def process_single_file(args):
    """Wrapper for multiprocessing with better error handling."""
    las_file, shp_path, location_name, polygon_name, overwrite = args
    
    try:
        date_match = re.match(r'^\d{8}', las_file.stem)
        date_str = date_match.group() if date_match else 'unknown'
        
        output_folder = OUTPUT_BASE_PATH / location_name / polygon_name
        pathout_base = output_folder / f"{date_str}_{polygon_name}"
        
        result = process_las_file(las_file, pathout_base, shp_path, location_name, overwrite)
        
        if result == "SKIPPED":
            return (True, las_file.name, "Skipped (already exists)")
        elif result is not None:
            n_polys = len(result['polygon_id'].unique())
            return (True, las_file.name, f"Processed {n_polys} polygons")
        else:
            return (False, las_file.name, "Returned None - check debug output above")
        
    except Exception as e:
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        return (False, las_file.name, error_msg)


def process_location(location_name, polygon_name, start_mop, end_mop, overwrite=False):
    """Process all LAS files for a location."""
    max_height = HEIGHTS.get(location_name, 30)
    
    print(f"\n{'='*70}")
    print(f"Processing: {location_name}")
    print(f"  Max height: {max_height}m")
    print(f"  DEM resolution: {DEM_RESOLUTION}m")
    print(f"  Elevation bins: {ELEVATION_BIN_SIZE}m (10cm)")
    print(f"  Slope window: {SLOPE_WINDOW} bins")
    print(f"  Using {N_PROCESSES} processes")
    print(f"{'='*70}")
    
    shp_path = BASE_SHAPE_PATH / polygon_name / f"{polygon_name}.shp"
    
    if not shp_path.exists():
        print(f"  ⚠️  Shapefile not found: {shp_path}")
        return
    
    las_folder = BASE_LAS_PATH / location_name / 'noveg'
    
    if not las_folder.exists():
        print(f"  ⚠️  LAS folder not found: {las_folder}")
        return
    
    las_files = sorted(las_folder.glob('*noveg.las'))
    
    if not las_files:
        print(f"  ⚠️  No LAS files found")
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
    
    successful = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/len(las_files):.1f}s per file)")
    print(f"  Success: {len(successful)}/{len(las_files)} files")
    
    if failed:
        print(f"\n  ❌ FAILED FILES ({len(failed)}):")
        for _, filename, msg in failed[:5]:
            print(f"    {filename}: {msg}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")


def main():
    """Main processing function."""
    print("\n" + "="*70)
    print("OPTIMIZED CLIFF SLOPE EXTRACTION - DEBUG MODE")
    print("Strategy: Top-Down DEM → Vertical Transect → Slopes")
    print("="*70)
    print(f"DEM Resolution: {DEM_RESOLUTION}m")
    print(f"Elevation Bins: {ELEVATION_BIN_SIZE}m")
    print(f"Min points per polygon: {MIN_POINTS_FOR_DEM}")
    print(f"CPU Cores: {N_CORES}, Using: {N_PROCESSES} processes")
    print("\nLocation-specific heights:")
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