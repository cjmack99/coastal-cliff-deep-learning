#!/usr/bin/env python3
"""
Optimized Cliff Slope Extraction: Top-Down DEM → Vertical Transects → Slopes

Strategy:
1. Create top-down DEM per polygon (interpolates/smooths points)
2. Extract vertical transects (cross-shore position vs elevation)
3. Compute local slopes from transects (matches your 10cm grid structure)

This combines the speed of DEMs with the vertical profile structure you need.
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
DEM_RESOLUTION = 0.25        # Top-down DEM resolution (m) - fine enough to capture cliff detail
ELEVATION_BIN_SIZE = 0.1     # Vertical bins (10 cm) - MATCHES YOUR EXISTING GRIDS
MIN_POINTS_FOR_DEM = 50      # Minimum points needed per polygon
SMOOTH_SIGMA = 0.5           # Gaussian smoothing for DEM (reduces noise)
SLOPE_WINDOW = 5             # Window size for slope calculation (in 10cm bins)

N_CORES = cpu_count()
N_PROCESSES = max(1, N_CORES // 4)


def create_topdown_dem(points_df, polygon_geom, resolution=DEM_RESOLUTION):
    """
    Step 1: Create traditional top-down DEM from points within polygon.
    
    Benefits:
    - Interpolates gaps
    - Smooths noise
    - Creates continuous surface
    
    Args:
        points_df: DataFrame with X, Y, Z
        polygon_geom: Shapely polygon
        resolution: Grid cell size (m)
    
    Returns:
        dem: 2D elevation array
        extent: (xmin, xmax, ymin, ymax)
        x_coords: 1D array of X coordinates
        y_coords: 1D array of Y coordinates
    """
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
    
    # Use linear interpolation (fast, good for dense LiDAR)
    dem = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # Optional: smooth to reduce noise (helps slope calculation)
    if SMOOTH_SIGMA > 0:
        valid_mask = ~np.isnan(dem)
        if valid_mask.any():
            dem_smooth = gaussian_filter(np.nan_to_num(dem, nan=0.0), sigma=SMOOTH_SIGMA)
            dem = np.where(valid_mask, dem_smooth, np.nan)
    
    return dem, (xmin, xmax, ymin, ymax), x_coords, y_coords


def extract_vertical_transect(dem, x_coords, y_coords, elevation_bins):
    """
    Step 2: Extract vertical transect from DEM.
    
    For each elevation bin, find the mean cross-shore position (X coordinate)
    where the DEM has that elevation. This is averaged across the Y dimension
    (alongshore) within the polygon.
    
    This matches your existing cube structure: one value per elevation bin per polygon.
    
    Args:
        dem: 2D elevation array (rows=Y, cols=X)
        x_coords: 1D array of X coordinates
        y_coords: 1D array of Y coordinates  
        elevation_bins: 1D array of elevation bin centers (e.g., [0.05, 0.15, 0.25, ...])
    
    Returns:
        transect_df: DataFrame with columns [z_bin, mean_x, std_x, n_cells]
    """
    if dem is None:
        return None
    
    # For each elevation bin, find which cells have that elevation
    results = []
    
    for z_center in elevation_bins:
        # Find cells within ±half bin of this elevation
        half_bin = ELEVATION_BIN_SIZE / 2
        mask = (dem >= z_center - half_bin) & (dem < z_center + half_bin)
        
        if not mask.any():
            continue
        
        # Get X coordinates of cells at this elevation
        # Since dem is (rows, cols) and rows=Y, cols=X:
        row_indices, col_indices = np.where(mask)
        x_values = x_coords[col_indices]
        
        n_cells = len(x_values)
        
        if n_cells >= 3:  # Need minimum number of cells
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
    """
    Step 3: Compute local slopes from vertical transect.
    
    Slope = dX/dZ (cross-shore distance change per elevation change)
    Then convert to angle: slope_angle = arctan(1/slope) since we want dZ/dX
    
    Uses moving window to compute slope at each elevation.
    
    Args:
        transect_df: DataFrame with z_bin, mean_x columns
        window_size: Number of bins for moving window (should be odd)
    
    Returns:
        transect_df with added columns: slope_magnitude, slope_angle_deg
    """
    if transect_df is None or len(transect_df) < 2:
        return None
    
    z_bins = transect_df['z_bin'].values
    x_values = transect_df['mean_x'].values
    
    slopes = np.full(len(z_bins), np.nan)
    half_window = window_size // 2
    
    for i in range(len(z_bins)):
        # Get window
        start_idx = max(0, i - half_window)
        end_idx = min(len(z_bins), i + half_window + 1)
        
        if end_idx - start_idx < 3:  # Need at least 3 points
            continue
        
        z_window = z_bins[start_idx:end_idx]
        x_window = x_values[start_idx:end_idx]
        
        # Fit linear slope: x = a*z + b, so dX/dZ = a
        if len(z_window) >= 2:
            coeffs = np.polyfit(z_window, x_window, 1)
            dx_dz = coeffs[0]  # Rate of cross-shore change per elevation
            
            # Convert to cliff face slope (dZ/dX)
            if abs(dx_dz) > 1e-6:  # Avoid division by zero
                dz_dx = 1.0 / dx_dz
                slopes[i] = abs(dz_dx)  # Magnitude of slope
    
    # Add to dataframe
    transect_df = transect_df.copy()
    transect_df['slope_magnitude'] = slopes
    transect_df['slope_angle_deg'] = np.degrees(np.arctan(slopes))
    
    return transect_df


def process_polygon(points_df, polygon_geom, polygon_id, max_height):
    """
    Complete processing pipeline for one polygon:
    1. Create top-down DEM
    2. Extract vertical transect
    3. Compute slopes
    
    Args:
        points_df: DataFrame with X, Y, Z
        polygon_geom: Shapely polygon geometry
        polygon_id: Polygon identifier
        max_height: Maximum elevation for this location
    
    Returns:
        DataFrame with columns: polygon_id, z_bin, mean_x, slope_magnitude, slope_angle_deg, n_cells
    """
    # Step 1: Create DEM
    dem, extent, x_coords, y_coords = create_topdown_dem(points_df, polygon_geom)
    
    if dem is None:
        return None
    
    # Step 2: Define elevation bins (matching your 10cm structure)
    # Use location-specific max_height
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
    """
    Process entire LAS file for all polygons.
    
    Args:
        pathin: Path to LAS file
        pathout_base: Base path for outputs
        polys: Path to polygon shapefile
        location_name: Location name (e.g., 'DelMar') - used to get max_height
        overwrite: Whether to overwrite existing files
    
    Outputs:
    - *_slopes.csv: Detailed data (polygon_id, z_bin, slope values)
    - *_slopes_grid.csv: Grid format matching your cubes (rows=polygons, cols=z_bins)
    - *_summary.csv: Per-polygon statistics
    """
    pathout_detail = Path(str(pathout_base) + '_slopes.csv')
    pathout_grid = Path(str(pathout_base) + '_slopes_grid.csv')
    pathout_summary = Path(str(pathout_base) + '_summary.csv')
    
    # Check if exists
    if pathout_detail.exists() and pathout_grid.exists() and not overwrite:
        print(f"Outputs exist, skipping: {pathout_base.name}")
        return
    
    print(f"\n--- Processing: {pathin.name} ---")
    
    # Get max height for this location
    max_height = HEIGHTS.get(location_name, 30)  # Default to 30 if not found
    print(f"  Max height: {max_height}m")
    
    # Read LAS
    with laspy.open(pathin) as lasf:
        las = lasf.read()
    
    print(f"  Loaded {len(las.x):,} points")
    
    # Load polygons FIRST to get CRS
    polys_gdf = gpd.read_file(polys)
    polys_gdf["Polygon_ID"] = polys_gdf.index
    print(f"  Loaded {len(polys_gdf)} polygons (CRS: {polys_gdf.crs})")
    
    # Create points dataframe using the polygon CRS
    df_pts = pd.DataFrame({
        'X': las.x,
        'Y': las.y,
        'Z': las.z
    })
    
    gdf_pts = gpd.GeoDataFrame(
        df_pts,
        geometry=[Point(x, y) for x, y in zip(las.x, las.y)],
        crs=polys_gdf.crs  # ← USE THE POLYGON CRS, DON'T HARDCODE
    )
    
    # Spatial join
    joined = gpd.sjoin(gdf_pts, polys_gdf[['Polygon_ID', 'geometry']],
                       how='inner', predicate='within')
    
    if len(joined) == 0:
        print("  No points in polygons, skipping")
        return None
    
    print(f"  {len(joined):,} points in {len(polys_gdf)} polygons")
    
    # Process each polygon
    all_results = []
    
    for poly_id in tqdm(range(len(polys_gdf)), desc="  Polygons", leave=False):
        poly_points = joined[joined['Polygon_ID'] == poly_id]
        
        if len(poly_points) < MIN_POINTS_FOR_DEM:
            continue
        
        poly_geom = polys_gdf.iloc[poly_id].geometry
        
        result = process_polygon(poly_points, poly_geom, poly_id, max_height)
        
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        print("  No valid results")
        return None
    
    # Concatenate all results
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Ensure output directory exists
    pathout_detail.parent.mkdir(parents=True, exist_ok=True)
    
    # Save detailed format
    df_all.to_csv(pathout_detail, index=False)
    
    # Create grid format (MATCHES YOUR CUBE STRUCTURE)
    slope_grid = df_all.pivot(
        index='polygon_id',
        columns='z_bin',
        values='slope_angle_deg'
    )
    slope_grid.to_csv(pathout_grid)
    
    # Create summary statistics per polygon
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


def process_single_file(args):
    """Wrapper for multiprocessing."""
    las_file, shp_path, location_name, polygon_name, overwrite = args
    
    try:
        date_match = re.match(r'^\d{8}', las_file.stem)
        date_str = date_match.group() if date_match else 'unknown'
        
        output_folder = OUTPUT_BASE_PATH / location_name / polygon_name
        pathout_base = output_folder / f"{date_str}_{polygon_name}"
        
        result = process_las_file(las_file, pathout_base, shp_path, location_name, overwrite)
        
        if result is not None:
            return (True, las_file.name, f"Processed {len(result['polygon_id'].unique())} polygons")
        return (False, las_file.name, "No valid data")
        
    except Exception as e:
        return (False, las_file.name, f"Error: {str(e)}")


def process_location(location_name, polygon_name, start_mop, end_mop, overwrite=False):
    """Process all LAS files for a location."""
    # Get max height for this location
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
    
    successful = sum(1 for r in results if r[0])
    
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/len(las_files):.1f}s per file)")
    print(f"  Success: {successful}/{len(las_files)} files")


def main():
    """Main processing function."""
    print("\n" + "="*70)
    print("OPTIMIZED CLIFF SLOPE EXTRACTION")
    print("Strategy: Top-Down DEM → Vertical Transect → Slopes")
    print("="*70)
    print(f"DEM Resolution: {DEM_RESOLUTION}m")
    print(f"Elevation Bins: {ELEVATION_BIN_SIZE}m (matches your existing grids)")
    print(f"Output Format: Grid (polygons × elevation_bins)")
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