import h5py
import numpy as np
import pyvista as pv
from rpca import rpca_on_scans

def visualize_rpca_vectors(scan_num, rpca_results, h5_path='HDF5_DATA/7_SCANS.h5'):
    """
    Visualizes the point cloud for a given scan along with its 3 RPCA principal vectors.
    """
    # 1. Fetch data
    key = f'{scan_num:04d}'
    try:
        with h5py.File(h5_path, 'r') as f:
            point_cloud = f[f'SCANS/{key}/POINT_CLOUD'][:]
        
        result = rpca_results[scan_num]
        center = result['center']
        components = result['components']
        variances = result['variances']
    except KeyError:
        print(f"Error: Scan {scan_num} not found.")
        return

    # 2. Initialize Plotter
    plotter = pv.Plotter()

    # Add the Point Cloud
    plotter.add_points(
        point_cloud, 
        color='lightblue', 
        point_size=4.0, 
        render_points_as_spheres=True, 
        opacity=0.6, 
        name='Sonar Points'
    )
    
    # Add the Robust Center as a distinct point
    plotter.add_points(
        np.array([center]), 
        color='red', 
        point_size=15.0, 
        render_points_as_spheres=True, 
        name='Robust Center'
    )

    # 3. Add the 3 Principal Vectors as Arrows
    colors = ['red', 'green', 'blue'] # Primary (1st), Secondary (2nd), Tertiary (3rd)
    
    # Scale arrows based on the bounding box of the point cloud so they are visible
    pts_range = np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0)
    scale_factor = np.max(pts_range) * 0.3 

    for i in range(3):
        # We use the component vector, scaled for visibility
        direction = components[i]
        
        arrow = pv.Arrow(
            start=center, 
            direction=direction, 
            scale=scale_factor, 
            shaft_radius=0.02, 
            tip_radius=0.06
        )
        
        plotter.add_mesh(
            arrow, 
            color=colors[i], 
            name=f'Component_{i+1}'
        )

    plotter.add_axes()
    print(f"Showing Scan {scan_num} | Red Arrow: 1st PC | Green: 2nd PC | Blue: 3rd PC")
    
    # Force Jupyter to let the 3D server start (using your existing backend)
    plotter.show(jupyter_backend='static')

def visualize_axis_selection(point_cloud, rpca_data, axis_index):
    """
      Visualizes the point cloud and the 3 principal components.
      The selected cylinder axis is highlighted in red.
    """
    center     = rpca_data['center']
    components = rpca_data['components']
    
    plotter = pv.Plotter()
    
    # 1. Plot the point cloud
    plotter.add_points(
        point_cloud, 
        color='lightblue', 
        point_size=5.0, 
        render_points_as_spheres=True, 
        opacity=0.6,
        name='Sonar Points'
    )
    
    # 2. Add the center point
    plotter.add_points(
        np.array([center]), 
        color='yellow', 
        point_size=15.0, 
        render_points_as_spheres=True
    )
    
    # Scale factor for the arrows (make them big enough to see)
    bounds = plotter.bounds
    scale  = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 0.3

    # 3. Plot the 3 PCA vectors
    for i, vec in enumerate(components):
        # Choose color and thickness based on whether it's the selected axis
        if i == axis_index:
            color = 'red'
            shaft_radius = 0.05
            label = f"Cylinder Axis ({i})"
        else:
            color = 'blue'
            shaft_radius = 0.02
            label = f"Cross-section Axis ({i})"
            
        arrow = pv.Arrow(start=center, direction=vec, scale=scale, shaft_radius=shaft_radius)
        plotter.add_mesh(arrow, color=color, name=label)

    # 4. Render
    plotter.add_axes()
    plotter.show(jupyter_backend='static')
