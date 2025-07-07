import pandas as pd
from geopy.distance import geodesic

# Constants for emissions (in kg per km)
DIESEL_EMISSION_FACTOR = 0.21
EV_EMISSION_FACTOR = 0.05

def load_delivery_data(path: str = "data/raw/delivery_five_cities.csv", nrows: int = 100000) -> pd.DataFrame:
    """
    Loads delivery data and converts GPS coordinates from microdegrees to decimal.

    Args:
        path (str): File path to the delivery CSV.
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Cleaned dataframe with lat/lng converted.
    """
    cols = ["poi_lat", "poi_lng", "receipt_lat", "receipt_lng"]
    try:
        df = pd.read_csv(path, usecols=cols, nrows=nrows)

        # Convert microdegree coordinates to decimal degrees
        for col in cols:
            df[col] = df[col] / 1e6

        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at: {path}")
    except Exception as e:
        raise Exception(f"Error reading delivery data: {e}")


def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes geodesic distance (in km) between origin and destination points.

    Args:
        df (pd.DataFrame): DataFrame with poi and receipt lat/lng.

    Returns:
        pd.DataFrame: DataFrame with added 'distance_km' column.
    """
    df["distance_km"] = df.apply(
        lambda row: geodesic(
            (row["poi_lat"], row["poi_lng"]),
            (row["receipt_lat"], row["receipt_lng"])
        ).km,
        axis=1
    )
    return df


def estimate_emissions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates CO₂ emissions and EV transition potential.

    Adds:
        - co2_kg: Diesel-based emissions.
        - suggest_ev: Boolean flag for trips <10 km.
        - ev_saving_kg: CO₂ saved if using EV.
        - ev_priority_score: Score for prioritizing EV conversion.

    Args:
        df (pd.DataFrame): DataFrame with distance info.

    Returns:
        pd.DataFrame: Enhanced with emission metrics.
    """
    df["co2_kg"] = df["distance_km"] * DIESEL_EMISSION_FACTOR
    df["suggest_ev"] = df["distance_km"] < 10
    df["ev_saving_kg"] = df["distance_km"] * (DIESEL_EMISSION_FACTOR - EV_EMISSION_FACTOR)

    def score_ev_priority(distance):
        if distance < 5:
            return 1.0
        elif distance < 10:
            return 0.9
        elif distance < 15:
            return 0.7
        elif distance < 20:
            return 0.5
        elif distance < 30:
            return 0.3
        else:
            return 0.1

    df["ev_priority_score"] = df["distance_km"].apply(score_ev_priority)

    return df
