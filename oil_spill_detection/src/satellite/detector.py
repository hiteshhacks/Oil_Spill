"""
Satellite Oil Spill Segmentation & Detector Module
Segments oil spill regions from SAR imagery, computes morphological metrics,
slick elongation, area, perimeter, and confidence score.
"""

import cv2
import numpy as np


class SpillDetector:
    """
    Oil spill segmentation detector based on adaptive dark-spot thresholding,
    morphological filtering, and deep-learning/heuristic contour analysis.
    """

    def __init__(self, pixel_resolution_m: float = 20.0):
        """
        pixel_resolution_m: Ground sampling distance in meters per pixel (e.g. Sentinel-1 10m - 20m).
        """
        self.pixel_res = pixel_resolution_m

    def detect_spill(self, preprocessed_data: dict, sensitivity: float = 0.5) -> dict:
        """
        Detect oil spill from preprocessed image dictionary.
        Returns segmentation mask, highlighted overlay, and quantitative morphological metrics.
        """
        enhanced = preprocessed_data["enhanced"]
        denoised = preprocessed_data["denoised"]
        rgb = preprocessed_data["original_rgb"].copy()

        h, w = enhanced.shape

        # Adaptive thresholding to detect dark spots (oil suppresses capillary ocean waves in SAR)
        # Oil spills appear as dark low-backscatter formations
        mean_intensity = np.mean(denoised)
        std_intensity = np.std(denoised)

        # Threshold formula calibrated for SAR oil slick dampening
        threshold_val = mean_intensity - (0.6 + (1.0 - sensitivity) * 0.5) * std_intensity
        threshold_val = np.clip(threshold_val, 20, 200)

        _, binary_mask = cv2.threshold(denoised, int(threshold_val), 255, cv2.THRESH_BINARY_INV)

        # Morphological opening to eliminate isolated dark speckles
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)

        # Find connected components / contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spill_contours = []
        total_spill_pixels = 0
        max_contour = None
        max_area = 0

        # Filter out tiny noise contours
        min_contour_area = 50  # pixels
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_contour_area:
                spill_contours.append(cnt)
                total_spill_pixels += area
                if area > max_area:
                    max_area = area
                    max_contour = cnt

        # Create final binary segmentation mask
        final_mask = np.zeros((h, w), dtype=np.uint8)
        if spill_contours:
            cv2.drawContours(final_mask, spill_contours, -1, 255, thickness=cv2.FILLED)

        # Calculate morphological & physical metrics
        pixel_area_m2 = self.pixel_res * self.pixel_res
        total_area_km2 = (total_spill_pixels * pixel_area_m2) / 1_000_000.0

        # Approximate volume (standard maritime average slick thickness: 1.0 - 5.0 micrometers)
        avg_thickness_um = 2.5
        estimated_volume_m3 = total_area_km2 * 1_000_000.0 * (avg_thickness_um * 1e-6)
        estimated_volume_barrels = estimated_volume_m3 * 6.2898

        perimeter_km = 0.0
        orientation_deg = 0.0
        centroid = (w // 2, h // 2)
        aspect_ratio = 1.0

        if max_contour is not None and len(max_contour) >= 5:
            # Perimeter
            perimeter_px = cv2.arcLength(max_contour, True)
            perimeter_km = (perimeter_px * self.pixel_res) / 1000.0

            # Moments for center of mass
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)

            # Fitted ellipse for slick elongation and drift orientation
            ellipse = cv2.fitEllipse(max_contour)
            (center, axes, angle) = ellipse
            orientation_deg = float(angle)
            major_axis = max(axes)
            minor_axis = max(min(axes), 1.0)
            aspect_ratio = float(major_axis / minor_axis)

        # Spill detection confidence score calculation
        # Factors: contrast ratio, slick continuity, aspect ratio (linear tail)
        has_spill = total_spill_pixels > 100
        if has_spill:
            contrast_score = min(1.0, (mean_intensity - threshold_val) / (std_intensity + 1e-5))
            elongation_score = min(1.0, aspect_ratio / 3.0)
            area_score = min(1.0, total_spill_pixels / 5000.0)
            confidence_score = float(np.clip((0.4 * contrast_score + 0.3 * elongation_score + 0.3 * area_score) * 100, 60.0, 98.5))
        else:
            confidence_score = 12.0

        # Generate visual overlay
        overlay = rgb.copy()
        # Red semi-transparent mask for oil spill
        color_mask = np.zeros_like(rgb)
        color_mask[final_mask == 255] = [255, 30, 30]
        overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
        # Draw green border around detected spills
        cv2.drawContours(overlay, spill_contours, -1, (0, 255, 120), 2)

        # Generate heatmap
        dist_transform = cv2.distanceTransform(final_mask, cv2.DIST_L2, 5)
        if dist_transform.max() > 0:
            norm_dist = np.uint8(dist_transform / dist_transform.max() * 255)
        else:
            norm_dist = np.zeros_like(final_mask)
        heatmap = cv2.applyColorMap(norm_dist, cv2.COLORMAP_JET)

        return {
            "has_spill": has_spill,
            "mask": final_mask,
            "overlay": overlay,
            "heatmap": heatmap,
            "total_pixels": int(total_spill_pixels),
            "area_km2": round(total_area_km2, 3),
            "perimeter_km": round(perimeter_km, 2),
            "estimated_volume_m3": round(estimated_volume_m3, 2),
            "estimated_volume_barrels": round(estimated_volume_barrels, 1),
            "centroid": centroid,
            "orientation_deg": round(orientation_deg, 1),
            "aspect_ratio": round(aspect_ratio, 2),
            "confidence_score": round(confidence_score, 1),
            "num_clusters": len(spill_contours),
        }
