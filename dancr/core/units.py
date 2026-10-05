"""Units and headers as people write them: 'inner area (IA) cm2' is an area, called IA, in cm²; a column in
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
    'inner area (IA) cm2' -> ('inner area', 'IA', 'cm2'); 'Pressure (bar)' -> ('Pressure', '', 'bar');
    'mass/area x 10000 (g/m2)' -> ('mass/area x 10000', '', 'g/m2'); 'outer area (OA)' -> ('outer area', 'OA', '')."""
    text = str(name).strip()
    unit = abbrev = ""
    words = text.split()
    if len(words) > 1 and (words[-1].lower() in _UNIT_OF or quantity_of(words[-1])) and (
            ")" in words[-2] or "]" in words[-2]):
        unit, text = words[-1], " ".join(words[:-1])             # 'inner area (IA) cm2': the unit after the brackets
    m = _UNIT_RE.search(text)
    if m:
        inside = m.group(1).strip()
        core = text[:m.start()].strip()
        initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", core)).lower()
        if inside.isalpha() and inside.isupper() and len(inside) >= 2 and (inside.lower() == initials or
                                                                           inside.lower() not in _UNIT_OF):
            abbrev = inside                                      # 'outer area (OA)': OA is its short name
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
    """'Pressure (bar)' -> 'bar'; 'Flow [m3/h]' -> 'm3/h'; 'inner area (IA) cm2' -> 'cm2'."""
    return header_parts(name)[2]


# ---------------------------------------------------------------- UCUM (interoperable unit codes)
# UCUM (http://unitsofmeasure.org) is the unit code system schema.org, Frictionless and most research tools
# understand. This maps the units people write in headers to their UCUM code, so a FAIR record is machine-
# readable rather than merely human-readable. Codes are case-sensitive; `to_ucum` returns "" for one it does
# not know (never a guess), and the caller keeps the raw unit beside it.
UCUM: dict[str, str] = {
    # mass
    "g": "g", "kg": "kg", "mg": "mg", "ug": "ug", "µg": "ug", "μg": "ug", "lb": "[lb_av]", "lbs": "[lb_av]",
    "oz": "[oz_av]", "t": "t", "tonne": "t", "tonnes": "t", "grams": "g", "gram": "g", "kilograms": "kg",
    # area
    "m2": "m2", "cm2": "cm2", "mm2": "mm2", "km2": "km2", "m²": "m2", "cm²": "cm2", "mm²": "mm2", "km²": "km2",
    "ha": "har", "hectares": "har", "acre": "[acr_us]", "acres": "[acr_us]", "ft2": "[ft_i]2", "in2": "[in_i]2",
    # length
    "m": "m", "cm": "cm", "mm": "mm", "km": "km", "um": "um", "µm": "um", "μm": "um", "nm": "nm", "in": "[in_i]",
    "ft": "[ft_i]", "mi": "[mi_i]", "inch": "[in_i]", "inches": "[in_i]", "feet": "[ft_i]", "metres": "m", "meters": "m",
    # volume
    "l": "L", "ml": "mL", "ul": "uL", "µl": "uL", "m3": "m3", "cm3": "cm3", "m³": "m3", "cm³": "cm3",
    "gal": "[gal_us]", "litres": "L", "liters": "L",
    # temperature
    "°c": "Cel", "°f": "[degF]", "ºc": "Cel", "ºf": "[degF]", "c": "Cel", "f": "[degF]",
    "celsius": "Cel", "fahrenheit": "[degF]", "degc": "Cel", "degf": "[degF]", "deg c": "Cel", "deg f": "[degF]",
    # duration
    "s": "s", "sec": "s", "secs": "s", "seconds": "s", "ms": "ms", "min": "min", "mins": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hours": "h", "days": "d",
    # pressure
    "pa": "Pa", "kpa": "kPa", "hpa": "hPa", "mpa": "MPa", "bar": "bar", "mbar": "mbar", "psi": "[psi]",
    "psia": "[psi]", "psig": "[psi]", "atm": "atm", "mmhg": "mm[Hg]",
    # speed
    "m/s": "m/s", "km/h": "km/h", "kmh": "km/h", "mph": "[mi_i]/h", "knots": "[kn_i]", "kn": "[kn_i]",
    # energy / power
    "j": "J", "kj": "kJ", "mj": "MJ", "kwh": "kW.h", "wh": "W.h", "cal": "cal", "kcal": "kcal",
    "w": "W", "kw": "kW", "mw": "MW",
    # concentration
    "ppm": "[ppm]", "ppb": "[ppb]", "mg/l": "mg/L", "mol/l": "mol/L", "mmol/l": "mmol/L", "g/l": "g/L",
    "mg/kg": "mg/kg", "µmol": "umol", "umol": "umol",
    # dimensionless
    "%": "%",
}


def to_ucum(unit: str) -> str:
    """The UCUM code for a unit as people write it ('g/m2' -> 'g/m2', '°C' -> 'Cel'), or "" when it is not known.
    A known quantity with an unknown scale is left to the caller rather than guessed."""
    u = (unit or "").strip().lower().replace("^", "").replace(" ", "")
    if not u:
        return ""
    if u in UCUM:
        return UCUM[u]
    if "/" in u and u.count("/") == 1:              # a per-unit like g/m2: map each side when both are known
        a, b = u.split("/")
        if a in UCUM and b in UCUM:
            return f"{UCUM[a]}/{UCUM[b]}"
    return ""
