"""
seasonality.py — Maximal Seasonality Feature Module
Institutional-Grade Version

Chunk 1:
    - Day-of-week seasonality
    - Month-of-year seasonality
    - Turn-of-month effect
    - End-of-month and end-of-quarter flows
    - Holiday proximity features
    - Intraday seasonality (if intraday data)
    - Rolling seasonal return averages
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# ============================================================
# 1. BASIC CALENDAR EXTRACTORS
# ============================================================

def extract_calendar_fields(df: pd.DataFrame):
    """
    Adds basic calendar fields:
        - day of week (0–6)
        - day of month (1–31)
        - month of year (1–12)
        - quarter (1–4)
    """
    dt = df.index
    df["DayOfWeek"] = dt.dayofweek
    df["DayOfMonth"] = dt.day
    df["Month"] = dt.month
    df["Quarter"] = dt.quarter
    return df


# ============================================================
# 2. DAY-OF-WEEK & MONTH-OF-YEAR ENCODINGS
# ============================================================

def add_day_of_week_features(df: pd.DataFrame):
    """
    One-hot encode day-of-week:
        Monday ... Sunday
    """
    dow = pd.get_dummies(df["DayOfWeek"], prefix="DOW")
    df[dow.columns] = dow
    return df


def add_month_features(df: pd.DataFrame):
    """
    One-hot encode month-of-year:
        Jan ... Dec
    """
    mo = pd.get_dummies(df["Month"], prefix="Month")
    df[mo.columns] = mo
    return df


# ============================================================
# 3. TURN-OF-MONTH EFFECT
# ============================================================

def turn_of_month_indicator(df: pd.DataFrame):
    """
    Turn-of-month effect:
        +1 if within first 3 trading days of month
        -1 if within last 3 trading days
         0 otherwise
    """
    day = df["DayOfMonth"]
    max_day = df.index.to_period("M").to_timestamp("M").day

    first_three = (day <= 3).astype(int)
    last_three = (day >= (max_day - 2)).astype(int)

    indicator = np.where(first_three == 1, 1,
                  np.where(last_three == 1, -1, 0))

    return indicator


def add_tom_feature(df: pd.DataFrame):
    df["TurnOfMonth"] = turn_of_month_indicator(df)
    return df


# ============================================================
# 4. END-OF-MONTH / END-OF-QUARTER FLOWS
# ============================================================

def end_of_month_feature(df: pd.DataFrame):
    """
    1 if last 2 trading days of month,
    0 otherwise.
    """
    dom = df["DayOfMonth"]
    month_ends = df.index.to_period("M").to_timestamp("M").day
    eom = (dom >= (month_ends - 1)).astype(int)
    return eom


def end_of_quarter_feature(df: pd.DataFrame):
    """
    1 if last 5 trading days of a quarter.
    """
    quarter_end_days = df.index.to_period("Q").to_timestamp("Q").day
    doq = df["DayOfMonth"]
    return (doq >= (quarter_end_days - 4)).astype(int)


def add_eom_eoq_features(df: pd.DataFrame):
    df["EOM"] = end_of_month_feature(df)
    df["EOQ"] = end_of_quarter_feature(df)
    return df


# ============================================================
# 5. HOLIDAY PROXIMITY EFFECTS
# ============================================================

def add_holiday_proximity(df: pd.DataFrame):
    """
    Adds:
        - Days until next holiday
        - Days since last holiday
    """
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())

    df["DaysToHoliday"] = df.index.map(
        lambda x: (holidays[holidays >= x].min() - x).days
                  if any(holidays >= x) else np.nan
    )

    df["DaysFromHoliday"] = df.index.map(
        lambda x: (x - holidays[holidays <= x].max()).days
                  if any(holidays <= x) else np.nan
    )
    return df


# ============================================================
# 6. INTRADAY SEASONALITY (if timestamp has hours)
# ============================================================

def add_intraday_features(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        df["Hour"] = np.nan
        df["Minute"] = np.nan
        return df

    df["Hour"] = df.index.hour
    df["Minute"] = df.index.minute

    # Hour-of-day dummy variables
    hour_dummies = pd.get_dummies(df["Hour"], prefix="H")
    df[hour_dummies.columns] = hour_dummies

    return df


# ============================================================
# 7. ROLLING SEASONAL RETURN AVERAGES
# ============================================================

def add_seasonal_return_stats(df: pd.DataFrame, windows=[5, 10, 20]):
    """
    Computes rolling average returns for:
        - each day of week
        - each month
    These act as historical drift priors.
    """
    rets = df["Close"].pct_change()

    for w in windows:
        df[f"DOWReturn_{w}"] = rets.groupby(df["DayOfWeek"]).rolling(w).mean().reset_index(level=0, drop=True)
        df[f"MonthReturn_{w}"] = rets.groupby(df["Month"]).rolling(w).mean().reset_index(level=0, drop=True)

    return df


# ============================================================
# 8. MASTER DISPATCHER (Chunk 1)
# ============================================================

def add_seasonality_features(df: pd.DataFrame):
    """
    Full seasonality block:
        - Day of week encoding
        - Month of year encoding
        - Turn of month indicator
        - End of month / quarter flows
        - Holiday proximity
        - Intraday seasonality
        - Seasonal rolling returns
    """
    df = extract_calendar_fields(df)
    df = add_day_of_week_features(df)
    df = add_month_features(df)
    df = add_tom_feature(df)
    df = add_eom_eoq_features(df)
    df = add_holiday_proximity(df)
    df = add_intraday_features(df)
    df = add_seasonal_return_stats(df)

    return df
