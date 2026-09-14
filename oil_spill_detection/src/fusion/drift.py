"""
Physics-Based Spill Drift and Trajectory Backtracking Module
Implements standard leeway drift physics (current velocity + 3% wind leeway with Coriolis deflection)
to back-project discharge origin and forecast future slick trajectory.
"""

from datetime import datetime, timedelta
import numpy as np


class DriftModel:
    """
    Physics-based ocean drift model for oil slick advection.
    Formula: V_drift = V_current + 0.03 * V_wind (rotated by Coriolis angle ~ 20-30 deg)
    """

    def __init__(self, wind_speed_knots: float = 12.0, wind_dir_deg: float = 225.0,
                 current_speed_knots: float = 1.2, current_dir_deg: float = 70.0):
        self.wind_speed = wind_speed_knots
        self.wind_dir = wind_dir_deg
        self.current_speed = current_speed_knots
        self.current_dir = current_dir_deg

    def compute_drift_velocity(self) -> tuple[float, float, float]:
        """
        Computes the net resultant drift vector (speed in knots, heading in degrees, u & v in km/h).
        Returns: (net_speed_knots, net_dir_deg, speed_kmh)
        """
        # Wind leeway factor = 3% of wind speed
        wind_leeway_speed = 0.03 * self.wind_speed

        # Coriolis deflection in Northern Hemisphere ~ +20 degrees to the right of wind
        effective_wind_dir = (self.wind_dir + 20.0) % 360.0

        # Current vector components (knots)
        curr_rad = np.radians(self.current_dir)
        u_curr = self.current_speed * np.sin(curr_rad)
        v_curr = self.current_speed * np.cos(curr_rad)

        # Wind vector components (knots)
        wind_rad = np.radians(effective_wind_dir)
        u_wind = wind_leeway_speed * np.sin(wind_rad)
        v_wind = wind_leeway_speed * np.cos(wind_rad)

        # Total drift vector
        u_total = u_curr + u_wind
        v_total = v_curr + v_wind

        net_speed_knots = np.sqrt(u_total**2 + v_total**2)
        net_dir_deg = np.degrees(np.arctan2(u_total, v_total)) % 360.0

        # 1 knot = 1.852 km/h
        speed_kmh = net_speed_knots * 1.852

        return float(net_speed_knots), float(net_dir_deg), float(speed_kmh)

    def forecast_drift(self, start_lat: float, start_lon: float,
                       hours: int = 48, step_hours: int = 3) -> list[dict]:
        """
        Forecast forward trajectory of oil slick over time for containment and emergency response.
        """
        _, drift_dir, speed_kmh = self.compute_drift_velocity()
        rad = np.radians(drift_dir)

        points = []
        current_time = datetime.now()

        for h in range(0, hours + 1, step_hours):
            distance_km = speed_kmh * h
            # Spread radius increases with square root of time (turbulent diffusion)
            spread_radius_km = 0.5 + 0.3 * np.sqrt(max(1, h))

            delta_lat = (distance_km * np.cos(rad)) / 111.0
            delta_lon = (distance_km * np.sin(rad)) / (111.0 * np.cos(np.radians(start_lat)))

            points.append({
                "hour": h,
                "timestamp": current_time + timedelta(hours=h),
                "lat": start_lat + delta_lat,
                "lon": start_lon + delta_lon,
                "distance_km": round(distance_km, 2),
                "spread_radius_km": round(spread_radius_km, 2)
            })

        return points

    def backtrack_spill_origin(self, detected_lat: float, detected_lon: float,
                               backtrack_hours: float = 3.5) -> dict:
        """
        Backtrack observed slick position to estimate origin location and time of discharge.
        """
        _, drift_dir, speed_kmh = self.compute_drift_velocity()
        # Backtrack is reverse direction
        reverse_rad = np.radians((drift_dir + 180.0) % 360.0)

        distance_km = speed_kmh * backtrack_hours
        delta_lat = (distance_km * np.cos(reverse_rad)) / 111.0
        delta_lon = (distance_km * np.sin(reverse_rad)) / (111.0 * np.cos(np.radians(detected_lat)))

        origin_lat = detected_lat + delta_lat
        origin_lon = detected_lon + delta_lon

        return {
            "origin_lat": round(origin_lat, 5),
            "origin_lon": round(origin_lon, 5),
            "drift_distance_km": round(distance_km, 2),
            "backtrack_hours": backtrack_hours,
            "net_drift_dir": round(drift_dir, 1),
            "net_drift_speed_knots": round(speed_kmh / 1.852, 2)
        }
