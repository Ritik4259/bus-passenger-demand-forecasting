"""Synthetic Trip-Level Dataset Generator for Bus Passenger Demand Forecasting.

Grounded in the Phagwara - Jalandhar - Lovely Professional University (LPU) NH-44 corridor in Punjab, India.

Generates realistic bus trips incorporating:
- Distinct university & regional route profiles with specific capacities and baseline demands
- Diurnal peak/off-peak student & commuter patterns
- Train arrival spikes (Phagwara Junction on Sunday/Monday)
- Weekend leisure & shopping surges (Model Town Jalandhar)
- Intra-campus class dismissals (Block 34 shuttle)
- Unconstrained latent passenger demand (target) alongside capacity-censored boardings
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RouteProfile:
    """Configuration profile for a transit route in the Punjab/LPU transit network."""

    route_id: str
    name: str
    default_capacity: int
    base_demand: float
    am_peak_multiplier: float
    pm_peak_multiplier: float
    weekend_multiplier: float
    rain_sensitivity: float
    origin: str
    destination: str
    departure_interval_minutes: int = 30


# Canonical transit corridors for the Phagwara - Jalandhar - LPU region
DEFAULT_ROUTES: List[RouteProfile] = [
    RouteProfile(
        route_id="LPU_JAL_101",
        name="LPU Main Gate - Jalandhar Bus Stand (via Rama Mandi)",
        default_capacity=50,
        base_demand=28.0,
        am_peak_multiplier=2.5,
        pm_peak_multiplier=2.7,
        weekend_multiplier=0.50,
        rain_sensitivity=1.25,
        origin="LPU Main Gate",
        destination="Jalandhar City Bus Stand",
        departure_interval_minutes=20,
    ),
    RouteProfile(
        route_id="PHG_LPU_202",
        name="Phagwara Junction Station - LPU Campus",
        default_capacity=40,
        base_demand=26.0,
        am_peak_multiplier=1.8,
        pm_peak_multiplier=1.9,
        weekend_multiplier=1.65,  # Spikes heavily Sunday evening & Monday morning for train arrivals
        rain_sensitivity=1.15,
        origin="Phagwara Junction Railway Station",
        destination="LPU Campus Uni-Mall",
        departure_interval_minutes=30,
    ),
    RouteProfile(
        route_id="LPU_MDL_303",
        name="LPU Campus - Model Town, Jalandhar",
        default_capacity=45,
        base_demand=24.0,
        am_peak_multiplier=1.1,
        pm_peak_multiplier=1.4,
        weekend_multiplier=1.45,  # Weekend and Friday evening leisure/shopping surge
        rain_sensitivity=1.05,
        origin="LPU Campus",
        destination="Model Town Market Jalandhar",
        departure_interval_minutes=30,
    ),
    RouteProfile(
        route_id="INT_CAMPUS_404",
        name="LPU Main Gate - Block 34 / Uni-Hospital (Campus Loop)",
        default_capacity=60,
        base_demand=34.0,
        am_peak_multiplier=2.4,
        pm_peak_multiplier=2.3,
        weekend_multiplier=0.25,  # Low on weekends
        rain_sensitivity=1.35,
        origin="LPU Main Gate 1",
        destination="Block 34 / Uni-Hospital",
        departure_interval_minutes=15,
    ),
    RouteProfile(
        route_id="JAL_ASR_505",
        name="Jalandhar Cantt - Phagwara GT Road Feeder",
        default_capacity=35,
        base_demand=16.0,
        am_peak_multiplier=1.6,
        pm_peak_multiplier=1.7,
        weekend_multiplier=0.70,
        rain_sensitivity=1.10,
        origin="Jalandhar Cantt Station",
        destination="Phagwara GT Road",
        departure_interval_minutes=45,
    ),
]


def _calculate_diurnal_factor(hour: float, route: RouteProfile) -> float:
    """Calculate the diurnal (time-of-day) demand multiplier.

    Uses smooth Gaussian-like peaks:
    - Intra-campus route: class transitions at 8.75 (8:45 AM), 13.0 (1:00 PM), 17.0 (5:00 PM)
    - General routes: AM peak (08:00) and PM peak (17:30)
    - Model Town route: evening recreation peak (17:00 - 21:00)
    """
    if hour < 5.0 or hour >= 23.5:
        return 0.15

    if route.route_id == "INT_CAMPUS_404":
        # Intra-campus class dismissals
        b1 = 1.8 * np.exp(-0.5 * ((hour - 8.75) / 0.8) ** 2)
        b2 = 1.6 * np.exp(-0.5 * ((hour - 13.0) / 0.9) ** 2)
        b3 = 2.0 * np.exp(-0.5 * ((hour - 17.0) / 0.9) ** 2)
        return max(0.15, float(0.40 + b1 + b2 + b3))

    if route.route_id == "LPU_MDL_303":
        # Evening leisure curve
        eve_bump = 2.1 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
        mid_bump = 0.6 * np.exp(-0.5 * ((hour - 13.0) / 2.0) ** 2)
        return max(0.2, float(0.35 + eve_bump + mid_bump))

    # Standard commuter / student corridor curve
    am_bump = np.exp(-0.5 * ((hour - 8.0) / 1.2) ** 2)
    pm_bump = np.exp(-0.5 * ((hour - 17.5) / 1.4) ** 2)
    midday_bump = 0.55 * np.exp(-0.5 * ((hour - 12.5) / 2.5) ** 2)

    factor = (
        0.40
        + (route.am_peak_multiplier - 1.0) * am_bump
        + (route.pm_peak_multiplier - 1.0) * pm_bump
        + midday_bump
    )
    return max(0.1, float(factor))


def _calculate_monthly_factor(month: int) -> float:
    """Seasonality multiplier across academic and monsoon months in Punjab (1-12)."""
    seasonal_curve = {
        1: 0.95,  # Winter break resumption
        2: 1.05,  # Full semester in session
        3: 1.08,  # Peak term
        4: 1.04,  # End term preparations
        5: 0.96,  # Exam dispersals
        6: 0.75,  # Summer vacation (low college demand)
        7: 0.88,  # Monsoon arrival / new admissions
        8: 1.12,  # Freshers registration & heavy student arrivals
        9: 1.10,  # Full autumn semester
        10: 1.08, # Youth fest / cultural events
        11: 1.05, # Mid-terms
        12: 0.85, # Winter holiday break
    }
    return seasonal_curve.get(month, 1.0)


def generate_synthetic_trips(
    start_date: str = "2026-01-01",
    num_days: int = 180,
    routes: Optional[List[RouteProfile]] = None,
    service_start_hour: int = 5,
    service_end_hour: int = 23,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic trip-level dataset grounded in the Punjab / LPU transit network."""
    rng = np.random.default_rng(random_state)
    active_routes = routes or DEFAULT_ROUTES

    start_dt = datetime.fromisoformat(start_date)
    records: List[Dict] = []
    trip_counter = 1

    # Pre-generate daily weather conditions in Punjab
    daily_weather: Dict[str, Dict] = {}
    for d in range(num_days):
        current_day = (start_dt + timedelta(days=d)).date()
        month = current_day.month
        # Monsoon season in Punjab (July-August) has higher rain probability
        rain_prob = 0.40 if month in (7, 8) else 0.20
        is_raining = bool(rng.uniform(0, 1) < rain_prob)

        # Seasonal temperature curve in Punjab: 10C in Jan to 38C in June
        base_temp = 12.0 + 26.0 * np.sin(np.pi * (month - 1) / 10.0)
        temp_c = float(rng.normal(loc=base_temp, scale=3.5))
        daily_weather[str(current_day)] = {
            "is_raining": is_raining,
            "temperature_c": round(temp_c, 1),
        }

    for day_offset in range(num_days):
        day_date = start_dt + timedelta(days=day_offset)
        day_str = str(day_date.date())
        weather_info = daily_weather[day_str]
        is_raining = weather_info["is_raining"]
        day_of_week = day_date.weekday()  # Monday = 0, Sunday = 6
        is_weekend = day_of_week in (5, 6)
        month = day_date.month

        # Occasional university fest or sports meet shock
        event_route_id = None
        if rng.uniform(0, 1) < 0.08:
            event_route_id = rng.choice([r.route_id for r in active_routes])

        for route in active_routes:
            interval = timedelta(minutes=route.departure_interval_minutes)
            current_trip_time = day_date.replace(
                hour=service_start_hour, minute=0, second=0, microsecond=0
            )
            end_trip_time = day_date.replace(
                hour=service_end_hour, minute=59, second=0, microsecond=0
            )

            # Route-specific day-of-week multipliers
            dow_factor = route.weekend_multiplier if is_weekend else 1.0

            # Special dynamic: Sunday night train arrival surge on PHG_LPU_202
            if route.route_id == "PHG_LPU_202" and day_of_week == 6:
                dow_factor = 2.1
            # Special dynamic: Friday evening leisure surge on LPU_MDL_303
            elif route.route_id == "LPU_MDL_303" and day_of_week == 4:
                dow_factor = 1.4

            monthly_factor = _calculate_monthly_factor(month)
            rain_factor = route.rain_sensitivity if is_raining else 1.0

            while current_trip_time <= end_trip_time:
                hour_float = current_trip_time.hour + current_trip_time.minute / 60.0
                diurnal_factor = _calculate_diurnal_factor(hour_float, route)

                # Special event multiplier
                is_event_trip = False
                event_mult = 1.0
                if event_route_id == route.route_id and 17.0 <= hour_float <= 21.0:
                    is_event_trip = True
                    event_mult = 1.70

                expected_demand = (
                    route.base_demand
                    * diurnal_factor
                    * dow_factor
                    * monthly_factor
                    * rain_factor
                    * event_mult
                )

                # Stochastic variance via negative binomial sampling
                dispersion_r = 15.0
                prob_nb = dispersion_r / (dispersion_r + expected_demand)
                sampled_demand = rng.negative_binomial(dispersion_r, prob_nb)
                latent_demand = max(0, int(sampled_demand))

                capacity = route.default_capacity
                boarded_passengers = min(latent_demand, capacity)
                occupancy_rate = latent_demand / capacity

                # Canonical operational action flag
                if occupancy_rate < 0.50:
                    action_flag = "UNDERUTILIZED"
                elif occupancy_rate <= 0.90:
                    action_flag = "OPTIMAL"
                else:
                    action_flag = "OVERFLOW_RISK"

                records.append(
                    {
                        "trip_id": f"TRIP_{trip_counter:07d}",
                        "route_id": route.route_id,
                        "route_name": route.name,
                        "origin": route.origin,
                        "destination": route.destination,
                        "departure_datetime": current_trip_time.isoformat(),
                        "capacity": capacity,
                        "is_weekend": int(is_weekend),
                        "day_of_week": day_of_week,
                        "hour": current_trip_time.hour,
                        "minute": current_trip_time.minute,
                        "month": month,
                        "is_raining": int(is_raining),
                        "temperature_c": weather_info["temperature_c"],
                        "is_special_event": int(is_event_trip),
                        "passenger_demand": latent_demand,
                        "boarded_passengers": boarded_passengers,
                        "occupancy_rate": round(occupancy_rate, 4),
                        "action_flag": action_flag,
                    }
                )
                trip_counter += 1
                current_trip_time += interval

    df = pd.DataFrame(records)
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"])
    df = df.sort_values(by=["departure_datetime", "route_id"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic bus trip demand dataset for Punjab/LPU corridor."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-01-01",
        help="Start date in YYYY-MM-DD format (default: 2026-01-01)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days to simulate (default: 180)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_trips.csv",
        help="Destination path for generated CSV dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.days} days of synthetic bus trips for LPU/Punjab corridors starting {args.start_date}..."
    )
    df = generate_synthetic_trips(
        start_date=args.start_date,
        num_days=args.days,
        random_state=args.seed,
    )

    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df):,} trips and saved to {output_path}")
    print("\nDataset Summary:")
    print(f"Total Trips: {len(df):,}")
    print(f"Routes: {df['route_id'].unique().tolist()}")
    print("Action Flag Distribution:")
    print(df["action_flag"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
