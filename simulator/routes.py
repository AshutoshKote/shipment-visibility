"""Predefined delivery routes as ordered (lat, lon) waypoints.

Set around the Singapore-Johor logistics corridor to match the target
company's domain. Coordinates are approximate and for simulation only.
"""

ROUTES = {
    "R-PORT-TUAS": {
        "name": "PSA Port -> Tuas Megaport",
        "waypoints": [
            (1.2644, 103.8400),  # Keppel / PSA terminal
            (1.2800, 103.8000),
            (1.3000, 103.7400),
            (1.3200, 103.6800),
            (1.3300, 103.6300),  # Tuas
        ],
    },
    "R-SIN-JHB": {
        "name": "Singapore CBD -> Johor Bahru (Woodlands crossing)",
        "waypoints": [
            (1.2830, 103.8510),  # CBD
            (1.3340, 103.8470),
            (1.3900, 103.8400),
            (1.4310, 103.7690),  # Woodlands Checkpoint
            (1.4640, 103.7660),  # JB Sentral
        ],
    },
    "R-CHANGI-JURONG": {
        "name": "Changi Air Cargo -> Jurong Industrial",
        "waypoints": [
            (1.3560, 103.9880),  # Changi
            (1.3500, 103.9300),
            (1.3380, 103.8500),
            (1.3300, 103.7600),
            (1.3290, 103.7180),  # Jurong
        ],
    },
    "R-WOODLANDS-SELETAR": {
        "name": "Woodlands -> Seletar Aerospace",
        "waypoints": [
            (1.4360, 103.7860),
            (1.4200, 103.8200),
            (1.4100, 103.8500),
            (1.4050, 103.8680),  # Seletar
        ],
    },
}

ROUTE_IDS = list(ROUTES.keys())

# Named facilities (origins / destinations) for geofencing.
# Geofence = within FACILITY_RADIUS_KM of one of these points.
FACILITIES = {
    "PSA Port": (1.2644, 103.8400),
    "Tuas Megaport": (1.3300, 103.6300),
    "Singapore CBD": (1.2830, 103.8510),
    "JB Sentral": (1.4640, 103.7660),
    "Changi Air Cargo": (1.3560, 103.9880),
    "Jurong Industrial": (1.3290, 103.7180),
    "Woodlands": (1.4360, 103.7860),
    "Seletar Aerospace": (1.4050, 103.8680),
}
