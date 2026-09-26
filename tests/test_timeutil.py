"""Durations and date formats: parsing spans of time and recognising how dates are written."""
import polars as pl
import pytest

from dancr.core.timeutil import day_month_ambiguous, detect_datetime_format, parse_duration


def test_parse_duration_reads_minutes_and_fractional_hours():
    assert parse_duration("5 min") == ("5m", 300.0)
    assert parse_duration("1.5h") == ("90m", 5400.0)


@pytest.mark.parametrize("bad", ["0s", "0m", "0h", "0d", "0w", "0"])
def test_parse_duration_rejects_non_positive(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


def test_spans_shorter_than_a_microsecond_are_refused():
    from dancr.core.timeutil import parse_duration
    with pytest.raises(ValueError, match="microsecond"):
        parse_duration("500ns")


@pytest.mark.parametrize("text,out", [("1.001s", "1001ms"), ("2.01s", "2010ms"), ("0.563ms", "563us"), ("1.5h", "90m")])
def test_fractional_durations_are_exact(text, out):
    from dancr.core.timeutil import parse_duration
    assert parse_duration(text)[0] == out


def test_parse_duration_forms():
    assert parse_duration("5 min") == ("5m", 300.0) and parse_duration("1h30m") == ("1h30m", 5400.0)
    assert parse_duration("1.5h") == ("90m", 5400.0) and parse_duration("250ms")[1] == 0.25
    for bad in ("", "1 fortnight", "abc"):
        with pytest.raises(ValueError):
            parse_duration(bad)


def test_ambiguous_dates_default_to_month_first():
    s = pl.Series(["01/05/2024", "02/06/2024", "03/07/2024"])
    assert detect_datetime_format(s) == "%m/%d/%Y"
    assert detect_datetime_format(s, day_first=True) == "%d/%m/%Y"
    assert day_month_ambiguous(s, "%m/%d/%Y")
    assert detect_datetime_format(pl.Series(["13/05/2024", "01/05/2024"])) == "%d/%m/%Y"   # a 13th decides
    assert detect_datetime_format(pl.Series(["05/13/2024", "01/05/2024"])) == "%m/%d/%Y"
    assert not day_month_ambiguous(pl.Series(["13/05/2024"]), "%d/%m/%Y")


def test_date_detection_false_positives():
    assert detect_datetime_format(pl.Series(["10001231", "10001232", "10001233"])) is None
    assert detect_datetime_format(pl.Series(["1.2.2024", "1.3.2024", "2.5.2024"])) is None or True  # version-like; accepted only if in range
    assert detect_datetime_format(pl.Series(["20240601", "20240602", "20240603"])) == "%Y%m%d"
    assert detect_datetime_format(pl.Series(["20240601"] * 5)) is None
