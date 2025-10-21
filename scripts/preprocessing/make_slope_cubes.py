#!/usr/bin/env python3
"""
Create Vertical Slope Profiles from LiDAR Point Clouds

This script processes LAS files and creates vertical slope profiles for each polygon.
Each polygon represents an alongshore slice, and we compute slope vertically using
elevation bins with a moving window approach.
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
from functools import partial
import time

# Suppress CRS warnings
import warnings
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

# Platform-specific base paths
BASE = ("/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
        if platform.system() == "Darwin"
        else "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs")

BASE_SHAPE_PATH = Path(os.path.join(BASE, "utilities", "shape_files"))
BASE_LAS_PATH = Path(os.path.join(BASE, "results"))
OUTPUT_BASE_PATH = Path(os.path.join(BASE, "results", "data_cubes", "slopes"))

# Processing parameters
VERTICAL_BIN_SIZE = 0.1  # 10 cm vertical bins
MAX_HEIGHT = 30.0  # maximum height to consider
WINDOW_SIZE = 3  # number of bins for moving window (centered)
MIN_POINTS_PER_BIN = 5  # minimum points needed in a bin

# Compute number of processes to use
N_CORES = cpu_count()
N_PROCESSES = max(1, N_CORES // 4)


def compute_vertical_slope(z_centers, z_values, window_size=3):
    """
    Compute slope at each vertical bin using a moving window.
    
    Args:
        z_centers: Array of bin center elevations
        z_values: Array of mean elevations in each bin
        window_size: Number of bins to use in window (must be odd)
    
    Returns:
        Array of slope values (same length as z_centers)
    """
    n = len(z_centers)
    slopes = np.full(n, np.nan)
    
    half_window = window_size // 2
    
    for i in range(n):
        # Get window indices
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        
        # Need at least 2 points to compute slope
        if end_idx - start_idx < 2:
            continue
        
        # Get window data
        z_win = z_centers[start_idx:end_idx]
        val_win = z_values[start_idx:end_idx]
        
        # Remove NaN values
        valid = ~np.isnan(val_win)
        if np.sum(valid) < 2:
            continue
        
        z_win = z_win[valid]
        val_win = val_win[valid]
        
        # Fit linear slope: dz/dh (change in position per change in height)
        # Using polyfit for simple linear regression
        if len(z_win) >= 2:
            coeffs = np.polyfit(z_win, val_win, 1)
            slopes[i] = abs(coeffs[0])  # slope magnitude
    
    return slopes


def makeGrid_with_slope(pathin, pathout_slope, polys, 
                        vertical_bin_size=VERTICAL_BIN_SIZE,
                        max_height=MAX_HEIGHT,
                        window_size=WINDOW_SIZE,
                        overwrite=False):
    """
    Reads in a LAS file and a shapefile of polygons, calculates vertical slope profile
    for each polygon using elevation bins with a moving window.
    
    Args:
        pathin: Path to input LAS file
        pathout_slope: Path to output CSV file for slope data
        polys: Path to shapefile with polygons
        vertical_bin_size: Size of vertical bins (default 0.1 m)
        max_height: Maximum height to consider (default 30 m)
        window_size: Number of bins for moving window (default 3)
        overwrite: Whether to overwrite existing files
    """
    
    # Check if output exists and skip if not overwriting
    if Path(pathout_slope).exists() and not overwrite:
        print(f"Output exists, skipping: {pathout_slope}")
        return
    
    print(f"\n--- Processing LAS: {pathin} ---")
    
    # Read LAS file
    with laspy.open(pathin) as lasf:
        las = lasf.read()
    
    print(f"Loaded {len(las.x):,} points")
    
    # Stack arrays (X, Y, Z)
    arr = np.vstack((las.x, las.y, las.z)).T
    
    # Load polygons
    polys_gdf = gpd.read_file(polys)
    polys_gdf["Polygon_ID"] = polys_gdf.index
    print(f"Loaded {len(polys_gdf)} polygons (alongshore slices)")
    
    # Create points GeoDataFrame
    df_pts = pd.DataFrame(arr, columns=['X', 'Y', 'Z'])
    gdf_pts = gpd.GeoDataFrame(
        df_pts,
        geometry=[Point(x, y) for x, y in zip(arr[:, 0], arr[:, 1])],
        crs=polys_gdf.crs
    )
    
    # Spatial join to assign each point to a polygon
    print("Performing spatial join...")
    joined = gpd.sjoin(gdf_pts, polys_gdf[['Polygon_ID', 'geometry']],
                       how='inner', predicate='within')
    
    print(f"{len(joined):,} points within polygons")
    
    if len(joined) == 0:
        print("No points within polygons, skipping...")
        return None
    
    # Create vertical bins
    z_bin_edges = np.arange(0, max_height + vertical_bin_size, vertical_bin_size)
    z_bin_centers = z_bin_edges[:-1] + vertical_bin_size / 2
    
    print(f"Computing vertical slope profiles with {window_size}-bin moving window...")
    print(f"Using {len(z_bin_centers)} vertical bins from 0 to {max_height} m")
    
    results = []
    
    # Process each polygon
    for poly_id in tqdm(range(len(polys_gdf)), desc="Processing polygons"):
        # Get points in this polygon
        poly_points = joined[joined['Polygon_ID'] == poly_id]
        
        if len(poly_points) < MIN_POINTS_PER_BIN:
            continue
        
        # Bin points by elevation
        z_values = poly_points['Z'].values
        x_values = poly_points['X'].values
        y_values = poly_points['Y'].values
        
        # For each vertical bin, compute mean cross-shore position (X)
        bin_indices = np.digitize(z_values, z_bin_edges) - 1
        
        # Calculate mean X position for each bin
        mean_x_per_bin = np.full(len(z_bin_centers), np.nan)
        points_per_bin = np.zeros(len(z_bin_centers), dtype=int)
        
        for i in range(len(z_bin_centers)):
            mask = bin_indices == i
            if np.sum(mask) >= MIN_POINTS_PER_BIN:
                mean_x_per_bin[i] = np.mean(x_values[mask])
                points_per_bin[i] = np.sum(mask)
        
        # Compute slope using moving window
        slopes = compute_vertical_slope(z_bin_centers, mean_x_per_bin, window_size)
        
        # Store results for each bin with valid slope
        for i, (z_center, slope, mean_x, n_pts) in enumerate(
            zip(z_bin_centers, slopes, mean_x_per_bin, points_per_bin)
        ):
            if not np.isnan(slope) and n_pts >= MIN_POINTS_PER_BIN:
                results.append({
                    'Polygon_ID': poly_id,
                    'z_bin': z_center,
                    'slope_magnitude': slope,
                    'slope_angle_deg': np.degrees(np.arctan(slope)),
                    'mean_x': mean_x,
                    'n_points': n_pts
                })
    
    # Convert to DataFrame and save
    if results:
        df_results = pd.DataFrame(results)
        
        # Ensure output directory exists
        Path(pathout_slope).parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed CSV
        df_results.to_csv(pathout_slope, index=False)
        print(f"\nSaved slope data to: {pathout_slope}")
        print(f"Computed slopes for {len(df_results)} polygon-bin combinations")
        
        # Also save as a grid (pivot table) to match your cube format
        grid_file = pathout_slope.with_name(pathout_slope.stem + '_grid.csv')
        slope_grid = df_results.pivot(index='Polygon_ID', columns='z_bin', values='slope_magnitude')
        slope_grid.to_csv(grid_file)
        print(f"Saved gridded format to: {grid_file}")
        print(f"Grid shape: {slope_grid.shape} (rows=polygons, cols=elevation bins)")
        
        # Print summary statistics
        print("\nSlope Statistics:")
        print(f"  Mean slope magnitude: {df_results['slope_magnitude'].mean():.3f}")
        print(f"  Median slope magnitude: {df_results['slope_magnitude'].median():.3f}")
        print(f"  Max slope magnitude: {df_results['slope_magnitude'].max():.3f}")
        print(f"  Mean slope angle: {df_results['slope_angle_deg'].mean():.1f}°")
        
        return df_results
    else:
        print("\nNo valid slopes computed (insufficient points in bins)")
        return None


def create_slope_visualization(df_results, polys_gdf, output_file, location_name, date_str):
    """
    Create visualization of vertical slope profiles.
    """
    if df_results is None or len(df_results) == 0:
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Example vertical profiles for a few polygons
    unique_polys = df_results['Polygon_ID'].unique()
    n_profiles = min(10, len(unique_polys))
    sample_polys = np.random.choice(unique_polys, n_profiles, replace=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    
    for i, poly_id in enumerate(sample_polys):
        poly_data = df_results[df_results['Polygon_ID'] == poly_id].sort_values('z_bin')
        ax1.plot(poly_data['slope_magnitude'], poly_data['z_bin'], 
                'o-', color=colors[i], alpha=0.7, linewidth=2, markersize=4,
                label=f'Polygon {poly_id}')
    
    ax1.set_xlabel('Slope Magnitude', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title(f'Example Vertical Slope Profiles\n(n={n_profiles} polygons)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    
    # Plot 2: Mean slope by elevation across all polygons
    mean_slope_by_z = df_results.groupby('z_bin')['slope_magnitude'].agg(['mean', 'std', 'count'])
    mean_slope_by_z = mean_slope_by_z[mean_slope_by_z['count'] >= 5]  # Only bins with enough data
    
    ax2.errorbar(mean_slope_by_z['mean'], mean_slope_by_z.index, 
                xerr=mean_slope_by_z['std'], fmt='o-', color='red', 
                linewidth=2, markersize=6, capsize=5, alpha=0.7)
    ax2.set_xlabel('Mean Slope Magnitude', fontsize=12)
    ax2.set_ylabel('Elevation (m)', fontsize=12)
    ax2.set_title(f'Mean Slope Profile (All Polygons)\n(n={len(unique_polys)} polygons)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Vertical Slope Analysis\n{location_name} - {date_str}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_file}")


def process_single_file(args):
    """
    Process a single LAS file. This function is designed to be called by multiprocessing.
    
    Args:
        args: tuple containing (las_file, shp_path, location_name, polygon_name, overwrite)
    
    Returns:
        tuple: (success, file_name, message)
    """
    las_file, shp_path, location_name, polygon_name, overwrite = args
    
    try:
        # Extract date from filename
        date_match = re.match(r'^\d{8}', las_file.stem)
        date_str = date_match.group() if date_match else 'unknown'
        
        # Create output paths
        output_folder = OUTPUT_BASE_PATH / location_name / polygon_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        pathout_slope = output_folder / f"{date_str}_{polygon_name}_slopes.csv"
        fig_file = output_folder / f"{date_str}_{polygon_name}_slopes.png"
        
        # Process the file
        df_results = makeGrid_with_slope(
            pathin=las_file,
            pathout_slope=pathout_slope,
            polys=shp_path,
            vertical_bin_size=VERTICAL_BIN_SIZE,
            max_height=MAX_HEIGHT,
            window_size=WINDOW_SIZE,
            overwrite=overwrite
        )
        
        # Create visualization if we got results
        if df_results is not None and len(df_results) > 0:
            # Load polygons for visualization
            polys_gdf = gpd.read_file(shp_path)
            polys_gdf["Polygon_ID"] = polys_gdf.index
            
            create_slope_visualization(df_results, polys_gdf, fig_file, 
                                     location_name, date_str)
        
        return (True, las_file.name, f"Successfully processed {len(df_results) if df_results is not None else 0} polygon-bin combinations")
        
    except Exception as e:
        error_msg = f"Error processing {las_file.name}: {str(e)}"
        return (False, las_file.name, error_msg)


def process_location(location_name, polygon_name, start_mop, end_mop, overwrite=False):
    """Process all LAS files for a single location using multiprocessing."""
    print(f"\n{'='*70}")
    print(f"Processing: {location_name} - {polygon_name}")
    print(f"Using {N_PROCESSES} processes ({N_CORES} cores available)")
    print(f"{'='*70}")
    
    # Construct shapefile path
    shp_path = BASE_SHAPE_PATH / polygon_name / f"{polygon_name}.shp"
    
    if not shp_path.exists():
        warnings.warn(f"Shapefile not found: {shp_path}\nSkipping...")
        return
    
    # Find all LAS files for this location
    las_folder = BASE_LAS_PATH / location_name / 'noveg'
    
    if not las_folder.exists():
        warnings.warn(f"LAS folder not found: {las_folder}\nSkipping...")
        return
    
    las_files = sorted(las_folder.glob('*noveg.las'))
    
    if not las_files:
        warnings.warn(f"No LAS files found in: {las_folder}\nSkipping...")
        return
    
    print(f"Found {len(las_files)} LAS files to process")
    
    # Prepare arguments for multiprocessing
    args_list = [(las_file, shp_path, location_name, polygon_name, overwrite) 
                 for las_file in las_files]
    
    # Process files in parallel
    start_time = time.time()
    
    with Pool(processes=N_PROCESSES) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc=f"Processing {location_name} files"
        ))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Summarize results
    successful = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    
    print(f"\n{'='*70}")
    print(f"LOCATION PROCESSING COMPLETE: {location_name}")
    print(f"{'='*70}")
    print(f"Total processing time: {processing_time:.1f} seconds")
    print(f"Average time per file: {processing_time/len(las_files):.1f} seconds")
    print(f"Successfully processed: {len(successful)}/{len(las_files)} files")
    
    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for success, filename, message in failed:
            print(f"  ❌ {filename}: {message}")
    
    if successful:
        print(f"\nSuccessful files ({len(successful)}):")
        for success, filename, message in successful[:5]:  # Show first 5
            print(f"  ✅ {filename}: {message}")
        if len(successful) > 5:
            print(f"  ... and {len(successful) - 5} more")
    
    print(f"\nCompleted processing for {location_name} - {polygon_name}")


def load_slope_data(location, polygon_name, date_str, base_path=None, as_grid=False):
    """
    Load slope data for a specific location, polygon, and date.
    
    Args:
        location: Location name (e.g., 'DelMar')
        polygon_name: Polygon identifier (e.g., 'DelMarPolygons595to620at10cm')
        date_str: Date string (e.g., '20170323')
        base_path: Base path to slope grids (optional)
        as_grid: If True, load the grid format; if False, load detailed format
    
    Returns:
        pd.DataFrame: DataFrame with slope data (grid or detailed format)
    """
    if base_path is None:
        base_path = OUTPUT_BASE_PATH
    else:
        base_path = Path(base_path)
    
    if as_grid:
        file_path = base_path / location / polygon_name / f"{date_str}_{polygon_name}_slopes_grid.csv"
    else:
        file_path = base_path / location / polygon_name / f"{date_str}_{polygon_name}_slopes.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Slope data file not found: {file_path}")
    
    return pd.read_csv(file_path, index_col=0 if as_grid else None)


def find_slope_grid_files(location, polygon_name, base_path=None):
    """
    Find all slope grid files for a location, similar to find_csv_files().
    Returns a sorted list of paths to all *_slopes_grid.csv files.
    
    Args:
        location: Location name (e.g., 'DelMar')
        polygon_name: Polygon identifier (e.g., 'DelMarPolygons595to620at10cm')
        base_path: Base path to slope grids (optional)
    
    Returns:
        list: Sorted list of file paths
    """
    if base_path is None:
        base_path = OUTPUT_BASE_PATH
    else:
        base_path = Path(base_path)
    
    folder = base_path / location / polygon_name
    
    if not folder.exists():
        return []
    
    # Find all grid files
    grid_files = sorted(folder.glob("*_slopes_grid.csv"))
    
    # Extract dates and sort
    date_re = re.compile(r'(\d{8})')
    dated_files = []
    for fp in grid_files:
        m = date_re.search(fp.name)
        if m:
            dated_files.append((m.group(1), str(fp)))
    
    dated_files.sort(key=lambda x: x[0])
    return [path for (_, path) in dated_files]


def load_slope_cube(location, polygon_name, base_path=None):
    """
    Load slope grids into a 3D NumPy array matching your cube format.
    
    Args:
        location: Location name (e.g., 'DelMar')
        polygon_name: Polygon identifier (e.g., 'DelMarPolygons595to620at10cm')
        base_path: Base path to slope grids (optional)
    
    Returns:
        tuple: (slope_cube, file_list) where slope_cube has shape (n_dates, n_polygons, n_z_bins)
    """
    file_list = find_slope_grid_files(location, polygon_name, base_path)
    
    if not file_list:
        raise ValueError(f"No slope grid files found for {location}/{polygon_name}")
    
    grids = []
    valid_files = []
    shapes = []
    
    for fp in file_list:
        df = pd.read_csv(fp, index_col=0)
        grids.append(df.values)
        shapes.append((fp, df.shape))
        valid_files.append(fp)
    
    if not grids:
        raise ValueError("No valid slope grid files found.")
    
    # Get reference shape
    ref_shape = shapes[0][1]
    
    # Filter out mismatched grids
    valid_grids = []
    valid_files_final = []
    mismatches = []
    
    for i, (fp, shape) in enumerate(shapes):
        if shape == ref_shape:
            valid_grids.append(grids[i])
            valid_files_final.append(fp)
        else:
            mismatches.append((fp, shape))
    
    if mismatches:
        print(f"\n⚠️ Omitting {len(mismatches)} files with mismatched grid shapes:")
        for fp, shape in mismatches:
            print(f"  {os.path.basename(fp)} — shape = {shape}, expected = {ref_shape}")
        print(f"\n✅ Using {len(valid_grids)} files with consistent shape {ref_shape}")
    
    if not valid_grids:
        raise ValueError("No files have consistent grid shapes.")
    
    slope_cube = np.stack(valid_grids, axis=0)
    print(f"Slope cube shape: {slope_cube.shape} (dates, polygons, elevation_bins)")
    
    return slope_cube, valid_files_final


def main():
    """Main processing function."""
    print("="*70)
    print("LiDAR Vertical Slope Profile Generator")
    print("="*70)
    print(f"Platform: {platform.system()}")
    print(f"Base path: {BASE}")
    print(f"Vertical bin size: {VERTICAL_BIN_SIZE} m")
    print(f"Moving window size: {WINDOW_SIZE} bins")
    print(f"Minimum points per bin: {MIN_POINTS_PER_BIN}")
    print(f"CPU cores available: {N_CORES}")
    print(f"Processes to use: {N_PROCESSES} (cores // 4)")
    
    overall_start_time = time.time()
    
    # Process each location
    for location_name, polygon_name, start_mop, end_mop in LOCATIONS:
        process_location(location_name, polygon_name, start_mop, end_mop, overwrite=False)
    
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    print("\n" + "="*70)
    print("ALL LOCATIONS PROCESSED!")
    print("="*70)
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Processed {len(LOCATIONS)} locations using {N_PROCESSES} parallel processes")
    print("="*70)


if __name__ == "__main__":
    main()