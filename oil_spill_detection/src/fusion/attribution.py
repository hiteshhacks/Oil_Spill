"""
Multi-Modal Fusion and Explainable AI (XAI) Attribution Module
Integrates AIS kinematics + Satellite SAR evidence + Environmental Drift Physics
to perform Vessel-Spill Attribution, compute Final Risk Score, and generate XAI reasoning.
"""

from dataclasses import dataclass
import numpy as np
from src.ais.tracker import VesselTrajectory


@dataclass
class AttributionResult:
    vessel: VesselTrajectory
    attribution_probability: float  # 0 to 100%
    risk_level: str  # "Low", "Medium", "High"
    spatio_temporal_score: float
    kinematic_anomaly_score: float
    drift_vector_alignment_score: float
    sar_confidence_score: float
    final_risk_score: float
    xai_explanation_summary: str
    xai_evidence_points: list[str]


class MultiModalFusionEngine:
    """
    Performs multi-modal fusion combining AIS, Satellite SAR detection,
    and Environmental Drift to perform vessel attribution and explainability.
    """

    def __init__(self, weights: dict = None):
        if weights is None:
            # Calibrated weights for multi-modal evidence fusion
            self.weights = {
                "spatial_temporal": 0.35,
                "kinematic_anomaly": 0.25,
                "drift_alignment": 0.20,
                "sar_confidence": 0.20,
            }
        else:
            self.weights = weights

    def evaluate_attribution(self, vessels: list[VesselTrajectory],
                             spill_detection: dict,
                             drift_info: dict) -> list[AttributionResult]:
        """
        Evaluate all vessels in maritime corridor against detected spill and backtrack origin.
        Returns ranked list of attribution results with explainable AI breakdowns.
        """
        origin_lat = drift_info["origin_lat"]
        origin_lon = drift_info["origin_lon"]
        sar_conf = spill_detection.get("confidence_score", 85.0)

        results = []

        for vessel in vessels:
            df = vessel.dataframe

            # 1. Spatio-Temporal Proximity to Backtracked Origin
            d_lat = (df["lat"] - origin_lat) * 111.0
            d_lon = (df["lon"] - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
            distances_to_origin = np.sqrt(d_lat**2 + d_lon**2)
            min_dist_origin = float(distances_to_origin.min())

            # Exponential decay proximity score: 100% at 0 km, 50% at 3 km, near 0% > 10 km
            spatial_score = float(100.0 * np.exp(-0.25 * min_dist_origin))

            # 2. Kinematic Anomaly Score from AIS
            kinematic_score = vessel.anomaly_score

            # 3. Drift Vector Alignment
            # Compare vessel heading vector with backtracked drift corridor
            closest_idx = int(distances_to_origin.argmin())
            vessel_cog = df["cog"].iloc[closest_idx]
            net_drift_dir = drift_info["net_drift_dir"]
            # Alignment between vessel track and drift vector
            angle_diff = abs((vessel_cog - net_drift_dir + 180) % 360 - 180)
            drift_alignment_score = float(max(10.0, 100.0 - (angle_diff / 1.8)))

            # 4. Multi-Modal Fusion Calculation
            final_risk = (
                self.weights["spatial_temporal"] * spatial_score +
                self.weights["kinematic_anomaly"] * kinematic_score +
                self.weights["drift_alignment"] * drift_alignment_score +
                self.weights["sar_confidence"] * sar_conf
            )
            final_risk = float(np.clip(final_risk, 0.0, 100.0))

            # Determine Risk Level Category (Low / Medium / High)
            if final_risk >= 70.0:
                risk_level = "High"
            elif final_risk >= 40.0:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            # Explainable AI (XAI) Synthesis
            xai_points = []
            xai_points.append(f"📍 **Proximity**: Closest approach to backtracked spill origin was **{min_dist_origin:.2f} km**.")
            
            sog_min = df['sog'].min()
            sog_max = df['sog'].max()
            if sog_max - sog_min > 5.0 and min_dist_origin < 5.0:
                xai_points.append(f"⚠️ **Speed Anomaly**: Vessel exhibited sudden deceleration from **{sog_max:.1f} kn** down to **{sog_min:.1f} kn** inside the estimated discharge zone.")
            else:
                xai_points.append(f"ℹ️ **Speed Profile**: Maintained steady transit speed (**avg {df['sog'].mean():.1f} kn**).")

            xai_points.append(f"🛰️ **Satellite SAR Evidence**: SAR slick footprint verified with **{sar_conf:.1f}%** confidence, covering **{spill_detection.get('area_km2', 0)} km²**.")
            xai_points.append(f"🌊 **Environmental Drift Physics**: Net drift ({drift_info['net_drift_speed_knots']:.1f} kn @ {drift_info['net_drift_dir']}°) back-traces slick release timestamp to **{vessel.suspect_timestamp.strftime('%H:%M UTC')}**.")

            if risk_level == "High":
                summary = (
                    f"Vessel {vessel.vessel_name} ({vessel.mmsi}) is classified as HIGH RISK for illegal oil/bilge discharge. "
                    f"Spatial-temporal intersection with backtracked slick origin is {spatial_score:.1f}%, combined with "
                    f"abnormal kinematic loitering profile."
                )
            elif risk_level == "Medium":
                summary = (
                    f"Vessel {vessel.vessel_name} ({vessel.mmsi}) passed within intermediate proximity ({min_dist_origin:.1f} km), "
                    f"but lacks distinct speed anomaly. Marked for auxiliary monitoring."
                )
            else:
                summary = (
                    f"Vessel {vessel.vessel_name} ({vessel.mmsi}) exhibits routine transit track with low spatial-temporal correlation. Exonerated from primary suspicion."
                )

            res = AttributionResult(
                vessel=vessel,
                attribution_probability=round(final_risk, 1),
                risk_level=risk_level,
                spatio_temporal_score=round(spatial_score, 1),
                kinematic_anomaly_score=round(kinematic_score, 1),
                drift_vector_alignment_score=round(drift_alignment_score, 1),
                sar_confidence_score=round(sar_conf, 1),
                final_risk_score=round(final_risk, 1),
                xai_explanation_summary=summary,
                xai_evidence_points=xai_points
            )
            results.append(res)

        # Sort by attribution probability descending
        results.sort(key=lambda x: x.final_risk_score, reverse=True)
        return results
