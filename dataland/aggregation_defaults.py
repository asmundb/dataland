"""Shared ECOCLIMAP categories and regional domain definitions."""

EARTH_RADIUS_M = 6_371_229.0

TRANSTABLE = {
    "sea": [1],
    "lake": [2, 3],
    "nature": list(range(4, 24)),
    "urban": list(range(24, 34)),
    "forest": list(range(7, 15)),
    "glacier": [6],
}

OUTPUT_DOMAINS = {
    "metcoop": {
        "area_id": "metcoop",
        "description": "MetCoOp domain",
        "shape": (1069, 949),
        "center": (17.5, 63.3),
        "resolution": 2500,
        "projection": {
            "proj": "lcc",
            "lat_1": 63.3,
            "lat_2": 63.3,
            "lat_0": 63.3,
            "lon_0": 15.,
            "R": 6371000.0,
            #"false_easting": 1060122.15845653,
            #"false_northing": 1332565.78014384,
            "units": "m",
        },
    },
    "carra2": {
        "area_id": "carra2",
        "description": "CARRA2 domain",
        "shape": (2869, 2869),
        "center": (-45.0, 84.0),
        "resolution": 2500,
        "projection": {
            "proj": "stere",
            "lat_0": 90,
            "lon_0": -30.,
            "R": 6371229.0,
            "units": "m",
        },
    },
    "carra2_sub":{
        "area_id": "carra2_sub",
        "description": "CARRA2 subdomain",
        "shape": (1246, 1638),
        "center": (-45.0, 84.0),
        "resolution": 2500,
        "projection": {
            "semi_major_axis": 6371229.0,
            "semi_minor_axis": 6371229.0,
            "inverse_flattening": 0.0,
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "grid_mapping_name": "polar_stereographic",
            "standard_parallel": 90.0,
            "straight_vertical_longitude_from_pole": -30.0,
            "false_easting": 172840.374543307,
            "false_northing": 645049.059394855,
            "latitude_of_projection_origin": 90.0,
            "units": "m",
        },
    }
}