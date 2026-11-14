"""Stadium coordinates and roof type, keyed by the nflverse schedule's stadium name.

roof: "dome" (fixed), "retractable", or "outdoors". The model treats retractable
roofs as closed: teams close them in bad weather, and when they're open the
weather is mild, so weather effects are near zero either way. nflverse's own
roof labels are blank for future retractable games and wrong for several
international venues, so this table is the source of truth.
"""
from __future__ import annotations

# name: (lat, lon, roof)
STADIUMS: dict[str, tuple[float, float, str]] = {
    "Acrisure Stadium": (40.4468, -80.0158, "outdoors"),
    "Allegiant Stadium": (36.0909, -115.1833, "dome"),
    "AT&T Stadium": (32.7473, -97.0945, "retractable"),
    "Bank of America Stadium": (35.2258, -80.8528, "outdoors"),
    "Bernabeu": (40.4531, -3.6883, "retractable"),
    "Caesars Superdome": (29.9511, -90.0812, "dome"),
    "Empower Field at Mile High": (39.7439, -105.0201, "outdoors"),
    "Estadio Banorte": (19.3029, -99.1505, "outdoors"),
    "EverBank Stadium": (30.3239, -81.6373, "outdoors"),
    "FC Bayern Munich Stadium": (48.2188, 11.6247, "outdoors"),
    "Ford Field": (42.3400, -83.0456, "dome"),
    "GEHA Field at Arrowhead Stadium": (39.0489, -94.4839, "outdoors"),
    "Gillette Stadium": (42.0909, -71.2643, "outdoors"),
    "Hard Rock Stadium": (25.9580, -80.2389, "outdoors"),
    "Highmark Stadium": (42.7738, -78.7870, "outdoors"),
    "Huntington Bank Field": (41.5061, -81.6995, "outdoors"),
    "Lambeau Field": (44.5013, -88.0622, "outdoors"),
    "Levi's Stadium": (37.4033, -121.9694, "outdoors"),
    "Lincoln Financial Field": (39.9008, -75.1675, "outdoors"),
    "Lucas Oil Stadium": (39.7601, -86.1639, "retractable"),
    "Lumen Field": (47.5952, -122.3316, "outdoors"),
    "M&T Bank Stadium": (39.2780, -76.6227, "outdoors"),
    "Maracana Stadium": (-22.9122, -43.2302, "outdoors"),
    "Melbourne Cricket Ground": (-37.8200, 144.9834, "outdoors"),
    "Mercedes-Benz Stadium": (33.7554, -84.4008, "retractable"),
    "MetLife Stadium": (40.8135, -74.0745, "outdoors"),
    "Nissan Stadium": (36.1665, -86.7713, "outdoors"),
    "Northwest Stadium": (38.9077, -76.8645, "outdoors"),
    "Paycor Stadium": (39.0955, -84.5161, "outdoors"),
    "Raymond James Stadium": (27.9759, -82.5033, "outdoors"),
    "Reliant Stadium": (29.6847, -95.4107, "retractable"),
    "SoFi Stadium": (33.9535, -118.3392, "dome"),
    "Soldier Field": (41.8623, -87.6167, "outdoors"),
    "Stade de France": (48.9245, 2.3602, "outdoors"),
    "State Farm Stadium": (33.5276, -112.2626, "retractable"),
    "Tottenham Hotspur Stadium": (51.6043, -0.0664, "outdoors"),
    "U.S. Bank Stadium": (44.9736, -93.2575, "dome"),
    "Wembley Stadium": (51.5560, -0.2796, "outdoors"),
}


def model_roof(name: str) -> str | None:
    """Roof value in the model's vocabulary: 'outdoors' or 'closed'/'dome'."""
    s = STADIUMS.get(name)
    if s is None:
        return None
    return {"dome": "dome", "retractable": "closed", "outdoors": "outdoors"}[s[2]]
