import openrouteservice
from openrouteservice import convert

def get_optimized_route(df, api_key):
    try:
        # Initialize OpenRouteService client
        client = openrouteservice.Client(key=api_key)

        # Extract coordinates for delivery points
        coords = df[["poi_lng", "poi_lat"]].values.tolist()

        if len(coords) < 2:
            return None, "Need at least 2 locations for route optimization."

        # Format coordinates for optimization
        route = client.directions(coords, profile='driving-car', format='geojson')

        return route, None
    except Exception as e:
        return None, str(e)
