"""Reading values as a person means them: numbers written as text, typed dates and times, time zones."""
from datetime import datetime

import polars as pl
import pytest

from dancr.core.dtypes import datetime_literal, text_to_number_expr


@pytest.mark.parametrize("text,value", [("1,200.5", 1200.5), ("1,5", 1.5), ("2,25", 2.25), ("1,500", 1500.0),
                                        ("1.234.567,5", 1234567.5), (" -3 ", -3.0), ("abc", None),
                                        ("inf", None), ("-Infinity", None), ("NaN", None)])
def test_text_is_read_as_the_number_a_person_means(text, value):
    got = pl.DataFrame({"s": [text]}).select(text_to_number_expr(pl.col("s")))["s"][0]
    assert got == value


def test_ambiguous_or_impossible_typed_dates_are_refused():
    with pytest.raises(ValueError, match="day/month or month/day"):
        datetime_literal("01/05/2024", pl.Datetime("us"))
    assert pl.select(datetime_literal("13/05/2024", pl.Datetime("us")))[0, 0] == datetime(2024, 5, 13)
    with pytest.raises(ValueError, match="doesn't exist"):
        datetime_literal("2024-03-31 01:30", pl.Datetime("us", "Europe/London"))
    assert pl.select(datetime_literal("2024-06-01 12:00+02:00", pl.Datetime("us")))[0, 0] == datetime(2024, 6, 1, 12)


def test_a_time_inside_a_clock_change_gap_is_blank_not_an_error():
    from dancr.core.dtypes import align_time_column
    df = pl.DataFrame({"t": [datetime(2024, 3, 31, 2, 30), datetime(2024, 3, 31, 4)]})
    out = df.select(align_time_column(pl.col("t"), pl.Datetime("us"), pl.Datetime("us", "Europe/Oslo")))
    assert out["t"].to_list()[0] is None and out["t"].to_list()[1] is not None


def test_offset_literal_keeps_its_instant_against_a_zoned_column():
    lit = pl.select(datetime_literal("2024-06-01T12:00:00+02:00", pl.Datetime("us", "Europe/Oslo"))).item()
    assert (lit.hour, lit.utcoffset().total_seconds()) == (12, 7200)
    naive = pl.select(datetime_literal("2024-06-01 12:00", pl.Datetime("us", "Europe/Oslo"))).item()
    assert naive.hour == 12 and naive.utcoffset().total_seconds() == 7200
    # a naive column has no zone: the literal's own wall time is used
    assert pl.select(datetime_literal("2024-06-01T12:00:00+02:00", pl.Datetime("ms"))).item() == datetime(2024, 6, 1, 12)
    assert pl.select(datetime_literal("2024-06-01", pl.Date)).dtypes == [pl.Datetime("us")]
