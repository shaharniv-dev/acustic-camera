import numpy as np

class SpatialMath:
    def __init__(self, fov_h_deg=75.0, width=640, height=480):
        """
        Initializes the acoustic-optical projection model.
        Precomputes the inverse mapping of the ULA cone of confusion (hyperbolas).
        """
        self.w = width
        self.h = height
        
        # Define the optical center
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        
        # Calculate focal lengths (assuming rectilinear projection and square pixels)
        fov_h_rad = np.radians(fov_h_deg)
        self.fx = self.cx / np.tan(fov_h_rad / 2.0)
        self.fy = self.fx  
        
        # Generate the static azimuth mapping for the image plane
        self.azimuth_grid = self._precompute_azimuth_grid()

    def _precompute_azimuth_grid(self):
        """
        Computes the corresponding acoustic azimuth (theta) for every pixel in the frame.
        Vectorized implementation of the inverse conic projection.
        """
        # Create a 2D grid of pixel coordinates
        x_idx, y_idx = np.meshgrid(np.arange(self.w), np.arange(self.h))
        
        # Normalize pixel coordinates
        u = (x_idx - self.cx) / self.fx
        v = (y_idx - self.cy) / self.fy
        
        # Inverse mapping: theta = arctan(u / sqrt(1 + v^2))
        theta_rad = -np.arctan(u / np.sqrt(1.0 + v**2))
        
        return np.degrees(theta_rad)

    def map_srp_to_image(self, angles_deg, srp_powers):
        """
        Projects the 1D SRP-PHAT spatial power spectrum onto the 2D image plane.
        
        Args:
            angles_deg: 1D array of evaluated azimuth angles. Must be strictly increasing.
            srp_powers: 1D array of acoustic power corresponding to the angles.
            
        Returns:
            A 2D numpy array (height, width) containing the projected acoustic heatmap.
        """
        # Interpolate the acoustic power onto the precomputed spatial grid
        heatmap_2d = np.interp(self.azimuth_grid, angles_deg, srp_powers)
        
        return heatmap_2d
    
    def get_uncertainty_curve(self, azimuth_deg, fov_v_deg=40, num_points=60):
        """
        Calculates the pixel coordinates of the uncertainty hyperbola for a given azimuth.
        Includes axis inversion to match the precomputed grid polarity.
        
        Args:
            azimuth_deg: The horizontal DOA peak.
            fov_v_deg: Vertical field of view bounds to draw.
            num_points: Resolution of the curve.
            
        Returns:
            A tuple of (x_pixels, y_pixels) as integer numpy arrays.
        """
        # Invert the angle to match the reversed optical axis mapping
        theta = np.radians(-azimuth_deg)
        
        # Generate array of unknown elevation angles
        phi = np.radians(np.linspace(-fov_v_deg, fov_v_deg, num_points))
        
        # Forward perspective projection of the cone's cross-section
        x_raw = self.cx + self.fx * (np.tan(theta) / np.cos(phi))
        y_raw = self.cy + self.fy * np.tan(phi)
        
        # Filter points out of frame bounds
        valid_mask = (x_raw >= 0) & (x_raw < self.w) & (y_raw >= 0) & (y_raw < self.h)
        
        x_px = np.round(x_raw[valid_mask]).astype(int)
        y_px = np.round(y_raw[valid_mask]).astype(int)
        
        return x_px, y_px