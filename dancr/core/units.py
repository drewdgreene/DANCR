"""Units and headers as people write them: 'Leaf area (LA) cm2' is a leaf area, called LA, in cm²; a column in
grams is a mass. Used to read column names (core.understand), to name columns in sentences (steps) and to know
what words a question may use for a column (core.ask). Pure standard library."""
from __future__ import annotations

import re

_UNIT_RE = re.compile(r"[\(\[]\s*([^\)\]]{1,12})\s*[\)\]]\s*$")
_BRACKET = re.compile(r"[\(\[]\s*([^\)\]]{1,12})\s*[\)\]]")

# a unit -> what it measures, so "mass" finds a column in grams and "area" one in cm²
QUANTITIES = {
    "mass": {"g", "kg", "mg", "ug", "µg", "μg", "lb", "lbs", "oz", "t", "tonne", "tonnes", "grams", "gram", "kilograms"},
    "area": {"m2", "cm2", "mm2", "km2", "ha", "ft2", "in2", "m²", "cm²", "mm²", "km²", "sq m", "sq ft", "acre", "acres",
             "hectares"},
    "length": {"m", "cm", "mm", "km", "um", "µm", "μm", "nm", "in", "ft", "mi", "inch", "inches", "feet", "metres",
               "meters"},
    "volume": {"l", "ml", "ul", "µl", "m3", "cm3", "m³", "cm³", "gal", "litres", "liters"},
    "temperature": {"°c", "°f", "c", "f", "degc", "degf", "deg c", "deg f", "celsius", "fahrenheit", "ºc", "ºf"},
    "duration": {"s", "sec", "secs", "ms", "min", "mins", "h", "hr", "hrs", "hours", "minutes", "seconds", "days"},
    "pressure": {"pa", "kpa", "hpa", "mpa", "bar", "mbar", "psi", "psia", "psig", "atm", "mmhg"},
    "speed": {"m/s", "km/h", "kmh", "mph", "knots", "kn"},
    "energy": {"j", "kj", "mj", "kwh", "wh", "cal", "kcal"},
    "power": {"w", "kw", "mw"},
    "concentration": {"ppm", "ppb", "mg/l", "mol/l", "mmol/l", "µmol", "umol", "g/l", "mg/kg"},
}
QUANTITY_WORDS = {"mass": ["mass", "weight"], "area": ["area"], "length": ["length"], "volume": ["volume"],
                  "temperature": ["temperature", "temp"], "duration": ["duration"], "pressure": ["pressure"],
                  "speed": ["speed"], "energy": ["energy"], "power": ["power"], "concentration": ["concentration"]}
_UNIT_OF = {u: q for q, us in QUANTITIES.items() for u in us}


def quantity_of(unit: str) -> str:
    """What a unit measures: 'g' -> 'mass', 'cm2' -> 'area', 'g/m2' -> 'mass per area'; '' when it is not known."""
    u = (unit or "").strip().lower().replace("^", "").replace(" ", "")
    if not u:
        return ""
    if u in _UNIT_OF:
        return _UNIT_OF[u]
    if "/" in u and u.count("/") == 1:
        a, b = u.split("/")
        qa, qb = _UNIT_OF.get(a, ""), _UNIT_OF.get(b, "")
        if qa and qb and qa != qb:
            return f"{qa} per {qb}"
    return ""


def header_parts(name: str) -> tuple[str, str, str]:
    """(name, abbreviation, unit) of a column header as people write it:
    'Leaf area (LA) cm2' -> ('Leaf area', 'LA', 'cm2'); 'Pressure (bar)' -> ('Pressure', '', 'bar');
    'm/LA x 10000 (g/m2)' -> ('m/LA x 10000', '', 'g/m2'); 'polygon area (PA)' -> ('polygon area', 'PA', '')."""
    text = str(name).strip()
    unit = abbrev = ""
    words = text.split()
    if len(words) > 1 and (words[-1].lower() in _UNIT_OF or quantity_of(words[-1])) and (
            ")" in words[-2] or "]" in words[-2]):
        unit, text = words[-1], " ".join(words[:-1])             # 'Leaf area (LA) cm2': the unit after the brackets
    m = _UNIT_RE.search(text)
    if m:
        inside = m.group(1).strip()
        core = text[:m.start()].strip()
        initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", core)).lower()
        if inside.isalpha() and inside.isupper() and len(inside) >= 2 and (inside.lower() == initials or
                                                                           inside.lower() not in _UNIT_OF):
            abbrev = inside                                      # 'polygon area (PA)': PA is its short name
        elif not unit:
            unit = inside
        text = core
    if not abbrev:
        b = _BRACKET.search(text)
        if b and b.group(1).strip().isalpha() and b.group(1).strip().isupper():
            abbrev = b.group(1).strip()
            text = (text[:b.start()] + text[b.end():]).strip()
    return " ".join(text.split()) or str(name), abbrev, unit


def unit_from_name(name: str) -> str:
    """'Pressure (bar)' -> 'bar'; 'Flow [m3/h]' -> 'm3/h'; 'Leaf area (LA) cm2' -> 'cm2'."""
    return header_parts(name)[2]
