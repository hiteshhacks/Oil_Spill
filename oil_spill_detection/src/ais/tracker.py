"""
AIS Tracking and Anomaly Detection Module
Handles vessel trajectory reconstruction, kinematic feature extraction,
and physics-based kinematic anomaly scoring for suspected bilge/oil dumping.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


@dataclass
class VesselTrajectory:
    mmsi: str
    vessel_name: str
    vessel_type: str
    callsign: str
    destination: str
    flag: str
    dataframe: pd.DataFrame
    anomaly_score: float
    anomaly_status: str  # "Low Anomaly" or "High Anomaly"
    anomaly_flags: list[str]
    suspect_timestamp: datetime
    suspect_location: tuple[float, float]  # (lat, lon)


class AISTracker:
    """Manages AIS ingestion, trajectory reconstruction, and vessel kinematic analysis."""

    def __init__(self, ref_lat: float = 18.922, ref_lon: float = 72.834):
        """Default reference coordinate centered near maritime channel (e.g. Mumbai offshore corridor)."""
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon

    def generate_synthetic_scenario(self, spill_center_lat: float, spill_center_lon: float, base_time: datetime = None) -> list[VesselTrajectory]:
        """
        Generate realistic multi-vessel scenario with:
        - 1 Primary Suspect Vessel (Crude Oil Tanker with sudden deceleration & loitering near spill)
        - 1 Minor Suspect Vessel (Chemical Tanker passing in adjacent corridor)
        - 2 Normal Transit Vessels (Container Ship & Bulk Carrier maintaining normal transit speed)
        """
        if base_time is None:
            base_time = datetime.now() - timedelta(hours=4)

        vessels = []

        # --- 1. PRIMARY SUSPECT VESSEL: "MT Ocean Titan" (Crude Tanker) ---
        # Starts 6 hours ago, approaches spill center, drops speed from 14.8 knots to 3.2 knots, makes sharp course deviation
        times = [base_time - timedelta(hours=6 - i * 0.5) for i in range(13)]
        lat_step = (spill_center_lat - (spill_center_lat - 0.25)) / 6
        lon_step = (spill_center_lon - (spill_center_lon - 0.30)) / 6

        records_suspect = []
        for idx, t in enumerate(times):
            if idx < 6:
                # Cruising towards spill
                lat = (spill_center_lat - 0.25) + idx * lat_step + np.random.normal(0, 0.001)
                lon = (spill_center_lon - 0.30) + idx * lon_step + np.random.normal(0, 0.001)
                sog = 14.5 + np.random.normal(0, 0.4)
                cog = 48.0 + np.random.normal(0, 1.0)
            elif idx == 6 or idx == 7:
                # Dumping / slow maneuvering right at spill origin
                lat = spill_center_lat + np.random.normal(0, 0.002)
                lon = spill_center_lon + np.random.normal(0, 0.002)
                sog = 3.1 + np.random.normal(0, 0.3)  # Massive speed drop
                cog = 115.0  # Sharp course shift
            else:
                # Accelerating away
                step_after = idx - 7
                lat = spill_center_lat + step_after * 0.03 + np.random.normal(0, 0.001)
                lon = spill_center_lon + step_after * 0.04 + np.random.normal(0, 0.001)
                sog = 12.0 + step_after * 0.5
                cog = 52.0

            records_suspect.append({
                "timestamp": t,
                "lat": lat,
                "lon": lon,
                "sog": max(0.0, round(sog, 2)),
                "cog": round(cog % 360, 1),
                "heading": round(cog % 360, 1),
                "nav_status": "Underway using Engine" if sog > 5 else "Restricted Maneuverability"
            })

        df_suspect = pd.DataFrame(records_suspect)
        v1 = self._analyze_trajectory(
            mmsi="412985321",
            vessel_name="MT Ocean Titan",
            vessel_type="Crude Oil Tanker",
            callsign="9V8421",
            destination="MUMBAI OFFSHORE",
            flag="Panama",
            df=df_suspect,
            spill_lat=spill_center_lat,
            spill_lon=spill_center_lon
        )
        vessels.append(v1)

        # --- 2. SECONDARY VESSEL: "MV Pacific Star" (Chemical Tanker) ---
        records_v2 = []
        for idx, t in enumerate(times):
            # Passes 6-8 km to the east at moderate constant speed
            lat = (spill_center_lat - 0.20) + idx * 0.035 + np.random.normal(0, 0.001)
            lon = (spill_center_lon + 0.08) + idx * 0.025 + np.random.normal(0, 0.001)
            sog = 11.2 + np.random.normal(0, 0.3)
            cog = 35.0 + np.random.normal(0, 1.5)
            records_v2.append({
                "timestamp": t,
                "lat": lat,
                "lon": lon,
                "sog": round(sog, 2),
                "cog": round(cog % 360, 1),
                "heading": round(cog % 360, 1),
                "nav_status": "Underway using Engine"
            })
        df_v2 = pd.DataFrame(records_v2)
        v2 = self._analyze_trajectory(
            mmsi="563048000",
            vessel_name="MV Pacific Star",
            vessel_type="Chemical Tanker",
            callsign="9V9032",
            destination="SINGAPORE",
            flag="Singapore",
            df=df_v2,
            spill_lat=spill_center_lat,
            spill_lon=spill_center_lon
        )
        vessels.append(v2)

        # --- 3. NORMAL TRANSIT: "CMA CGM Everest" (Container Ship) ---
        records_v3 = []
        for idx, t in enumerate(times):
            lat = (spill_center_lat - 0.35) + idx * 0.06
            lon = (spill_center_lon - 0.15) + idx * 0.03
            sog = 19.5 + np.random.normal(0, 0.2)  # Fast constant transit
            cog = 28.0
            records_v3.append({
                "timestamp": t,
                "lat": lat,
                "lon": lon,
                "sog": round(sog, 2),
                "cog": round(cog % 360, 1),
                "heading": round(cog % 360, 1),
                "nav_status": "Underway using Engine"
            })
        df_v3 = pd.DataFrame(records_v3)
        v3 = self._analyze_trajectory(
            mmsi="228389600",
            vessel_name="CMA CGM Everest",
            vessel_type="Container Ship",
            callsign="FMCV",
            destination="ROTTERDAM",
            flag="France",
            df=df_v3,
            spill_lat=spill_center_lat,
            spill_lon=spill_center_lon
        )
        vessels.append(v3)

        # --- 4. NORMAL TRANSIT: "Bulk Prosperity" (Bulk Carrier) ---
        records_v4 = []
        for idx, t in enumerate(times):
            lat = (spill_center_lat + 0.25) - idx * 0.04
            lon = (spill_center_lon + 0.20) - idx * 0.03
            sog = 13.0 + np.random.normal(0, 0.3)
            cog = 215.0
            records_v4.append({
                "timestamp": t,
                "lat": lat,
                "lon": lon,
                "sog": round(sog, 2),
                "cog": round(cog % 360, 1),
                "heading": round(cog % 360, 1),
                "nav_status": "Underway using Engine"
            })
        df_v4 = pd.DataFrame(records_v4)
        v4 = self._analyze_trajectory(
            mmsi="354921000",
            vessel_name="Bulk Prosperity",
            vessel_type="Bulk Carrier",
            callsign="3FGT",
            destination="COLOMBO",
            flag="Panama",
            df=df_v4,
            spill_lat=spill_center_lat,
            spill_lon=spill_center_lon
        )
        vessels.append(v4)

        return vessels

    def _analyze_trajectory(self, mmsi: str, vessel_name: str, vessel_type: str,
                            callsign: str, destination: str, flag: str,
                            df: pd.DataFrame, spill_lat: float, spill_lon: float) -> VesselTrajectory:
        """Computes anomaly score based on speed drops, course changes, and spatial-temporal proximity."""
        anomaly_flags = []
        anomaly_score = 0.0

        # Calculate distances to spill center (in km using Haversine approximation)
        d_lat = (df["lat"] - spill_lat) * 111.0
        d_lon = (df["lon"] - spill_lon) * 111.0 * np.cos(np.radians(spill_lat))
        distances_km = np.sqrt(d_lat**2 + d_lon**2)
        df["dist_to_spill_km"] = distances_km

        min_dist_idx = int(distances_km.argmin())
        min_distance = float(distances_km.iloc[min_dist_idx])
        closest_point = (float(df["lat"].iloc[min_dist_idx]), float(df["lon"].iloc[min_dist_idx]))
        closest_time = df["timestamp"].iloc[min_dist_idx]

        # 1. Proximity score
        if min_distance < 2.0:
            anomaly_score += 35.0
            anomaly_flags.append(f"Critical proximity to spill origin ({min_distance:.2f} km)")
        elif min_distance < 8.0:
            anomaly_score += 15.0
            anomaly_flags.append(f"Moderate proximity to spill zone ({min_distance:.2f} km)")

        # 2. Kinematic Speed Drop Anomaly
        sog_std = df["sog"].std()
        sog_min = df["sog"].min()
        sog_max = df["sog"].max()
        speed_drop_ratio = (sog_max - sog_min) / (sog_max + 1e-5)

        if speed_drop_ratio > 0.6 and min_distance < 5.0:
            anomaly_score += 35.0
            anomaly_flags.append(f"Severe speed drop ({sog_max:.1f} kn -> {sog_min:.1f} kn) near spill")
        elif sog_std > 3.0:
            anomaly_score += 15.0
            anomaly_flags.append("Erratic speed variation")

        # 3. Course / Loitering Anomaly
        cog_diffs = np.abs(np.diff(df["cog"]))
        cog_diffs = np.minimum(cog_diffs, 360 - cog_diffs)
        max_turn = cog_diffs.max() if len(cog_diffs) > 0 else 0

        if max_turn > 50.0 and min_distance < 5.0:
            anomaly_score += 20.0
            anomaly_flags.append(f"Abrupt course change ({max_turn:.1f}°) within discharge zone")

        # 4. Vessel Type Risk Factor
        if "Tanker" in vessel_type:
            anomaly_score += 10.0
            anomaly_flags.append("High-risk cargo carrier (Tanker/Bunker)")

        anomaly_score = float(np.clip(anomaly_score, 5.0, 96.0))
        status = "High Anomaly" if anomaly_score >= 60.0 else "Low Anomaly"

        return VesselTrajectory(
            mmsi=mmsi,
            vessel_name=vessel_name,
            vessel_type=vessel_type,
            callsign=callsign,
            destination=destination,
            flag=flag,
            dataframe=df,
            anomaly_score=round(anomaly_score, 1),
            anomaly_status=status,
            anomaly_flags=anomaly_flags,
            suspect_timestamp=closest_time,
            suspect_location=closest_point
        )
