import cv2 as cv
import numpy as np

class OverlayRenderer:
    def __init__(self, mapper, alpha=0.6, colormap=cv.COLORMAP_JET, threshold_ratio=0.75):
        """
        Processes and blends the acoustic power matrix onto the camera frame.
        Includes On-Screen Display (OSD) for angle output and ambiguity guide line.
        """
        self.mapper = mapper
        self.alpha = alpha
        self.colormap = colormap
        self.threshold_ratio = threshold_ratio
        
        # OSD Style Parameters
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.text_color = (255, 255, 255) # White
        self.text_thickness = 2
        self.guide_color = (0, 255, 255)  # Cyan guide line
        self.guide_thickness = 2

    def render(self, frame, heatmap, angles, powers):
        """
        Renders heatmap, OSD text, and ambiguity guide line onto the frame.
        """
        # Normalize to standard 8-bit image format
        heatmap_norm = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        # Zero out low-energy values to filter noise and sidelobes
        _, mask = cv.threshold(heatmap_norm, int(255 * self.threshold_ratio), 255, cv.THRESH_TOZERO)
        
        # Apply the chosen colormap
        colored_heatmap = cv.applyColorMap(mask, self.colormap)
        
        # Perform C++ optimized blending on the entire frame
        full_blend = cv.addWeighted(frame, 1 - self.alpha, colored_heatmap, self.alpha, 0)
        
        # Create a 2D boolean mask for valid energy regions
        valid_mask = mask > 0
        
        # Blend only where acoustic data exists using vectorized numpy logic
        blended = np.where(valid_mask[..., None], full_blend, frame)

        # Draw OSD and Uncertainty Curve if valid acoustic data is available
        if powers is not None and angles is not None:
            max_idx = np.argmax(powers)
            max_angle = angles[max_idx]
            
            # Fetch hyperbola coordinates from SpatialMath
            x_curve, y_curve = self.mapper.get_uncertainty_curve(max_angle, fov_v_deg=40, num_points=60)
            
            if len(x_curve) > 2:
                # Format points for cv.polylines (requires (N, 1, 2) shape)
                pts = np.vstack((x_curve, y_curve)).T.reshape((-1, 1, 2))
                cv.polylines(blended, [pts], isClosed=False, 
                             color=self.guide_color, thickness=self.guide_thickness, lineType=cv.LINE_AA)

            # Format and draw text output
            text = f"Azimuth: {max_angle:.1f} deg"
            text_pos = (30, blended.shape[0] - 30)
            
            # Draw text with a thin black outline for better readability
            cv.putText(blended, text, text_pos, self.font, self.font_scale, 
                       (0, 0, 0), self.text_thickness + 2, cv.LINE_AA)
            cv.putText(blended, text, text_pos, self.font, self.font_scale, 
                       self.text_color, self.text_thickness, cv.LINE_AA)

        return blended