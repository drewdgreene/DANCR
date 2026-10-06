"""Tunables, choice lists and result types for the answer recipes."""
from __future__ import annotations

import copy
import json
import math
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from ..planner import Plan, PlanStep
from ..understand import (DataModel, Table, Column, Relation, ID, TEXT, SERIES, LOOKUP, bucket_for, norm, name_words)
from ..timeutil import parse_bucket


RULES_VERSION = 5           # bump when a change to these rules would build a different plan from the same spec

STAT_WORDS = {"sum": "Total", "mean": "Average", "count": "Number of rows", "max": "Highest", "min": "Lowest",
              "median": "Median", "std": "Standard deviation of"}
STAT_CHOICES = ["sum", "mean", "median", "min", "max", "std", "count"]
EVERY_CHOICES = ["1m", "15m", "1h", "1d", "1w", "1mo", "1q", "1y"]
EVERY_WORDS = {"1s": "second", "10s": "10 seconds", "30s": "30 seconds", "1m": "minute", "5m": "5 minutes",
               "15m": "15 minutes", "30m": "30 minutes", "1h": "hour", "6h": "6 hours", "1d": "day", "1w": "week",
               "1mo": "month", "1q": "quarter", "1y": "year", "1ms": "millisecond", "10ms": "10 ms", "50ms": "50 ms",
               "100ms": "100 ms", "500ms": "half second", "5s": "5 seconds"}
TOP_CHOICES = [5, 10, 20, 50]

# the recipes, in the order that breaks ties
RECIPES = ["compare", "groups", "trend", "breakdown", "top", "toprows", "relationship", "gaps", "outliers", "single",
           "distribution", "linked", "stacked", "rows", "describe", "change", "explain", "drivers", "forecast", "quality",
           "nearest", "map", "density", "place"]
WEIGHT = {"compare": 100, "groups": 62, "trend": 95, "breakdown": 90, "top": 75, "relationship": 60, "gaps": 65,
          "outliers": 55, "single": 30, "distribution": 45, "linked": 50, "stacked": 60, "rows": 25, "toprows": 40,
          "describe": 20, "change": 88, "explain": 86, "drivers": 68, "forecast": 58, "quality": 18,
          "nearest": 72, "map": 56, "density": 46, "place": 66}
NEAR_DEFAULT = "10km"       # a proximity match with no distance asked for: near enough to mean something
EXPERIMENT_ROWS = 5_000     # a table this small, with groups and no dates, is a study: its groups are compared first
TEST_CHOICES = [("auto", "Test: chosen for me"), ("welch", "Welch's t-test"), ("student", "Student's t-test"),
                ("rank", "Rank test (Mann–Whitney)")]
GROUP_MAX = 12              # a group with more values than this is a "top N" question rather than a breakdown
MAX_SUGGESTIONS = 8


class PlanError(ValueError):
    """A question that cannot be answered from these tables, with the reason in plain English."""


@dataclass
class Suggestion:
    spec: dict[str, Any]
    title: str
    recipe: str
    score: float
    why: str = ""
    view: str = "chart"

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec, "title": self.title, "recipe": self.recipe, "score": self.score,
                "why": self.why, "view": self.view}



AMOUNT_WORDS = {"sales", "sale", "revenue", "amount", "amounts", "cost", "costs", "spend", "spent", "profit", "income",
                "qty", "quantity", "quantities", "units", "unit", "count", "counts", "total", "sum", "volume", "orders",
                "items", "visits", "hours", "minutes", "calls", "tickets", "turnover", "paid", "payment", "payments",
                "sold", "bookings", "downloads", "clicks", "views", "impressions", "rainfall", "precipitation", "energy"}
READING_WORDS = {"tmax", "tmin", "tavg", "dewpoint", "dew", "wind", "windspeed", "gust", "kmh", "mph", "lat", "lon",
                 "salary", "salaries", "wage", "wages", "pay", "battery", "bounce", "duration", "time", "temperature", "temp", "pressure", "humidity", "speed", "velocity", "level", "depth", "height",
                 "voltage", "current", "rate", "ratio", "percent", "pct", "percentage", "price", "score", "age", "ph",
                 "conductivity", "salinity", "concentration", "density", "flow", "rating", "latitude", "longitude",
                 "lat", "lon", "lng", "altitude", "elevation", "weight", "mass", "size", "length", "width", "psi",
                 "psia", "bar", "tension", "load", "signal", "strength", "frequency", "reading", "average",
                 "mean", "median", "index", "margin", "utilisation", "utilization", "occupancy", "efficiency"}
MONEY_WORDS = {"amount", "sales", "revenue", "turnover", "income", "cost", "costs", "spend", "profit", "paid",
               "payment", "payments", "debit", "credit", "balance", "value", "total"}
CURRENCY_UNITS = {"$", "€", "£", "¥", "usd", "eur", "gbp", "jpy", "chf", "aud", "cad", "nok", "sek", "dkk", "k$", "m$"}


PART_WORDS = {"hour": "hour of the day", "weekday": "day of the week", "month": "month of the year", "day": "day of the month"}

QUANTITY_NAMES = (READING_WORDS | AMOUNT_WORDS | {"length", "width", "height", "weight", "mass", "area", "volume", "depth",
                                                   "diameter", "radius", "age", "size", "cost", "price"}) - {"total", "value", "reading"}
PAIRED_WORDS = {"before", "after", "pre", "post", "baseline", "follow", "followup", "week", "day", "month", "visit",
                "time", "t0", "t1", "t2", "start", "end", "initial", "final", "first", "second", "trial", "run"}



CELL_CHOICES = ("0.01", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "5", "10")
NEAR_CHOICES = ("1km", "5km", "10km", "25km", "50km", "100km")
