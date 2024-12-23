import pandas as pd
import numpy as np

class TimeFeature:
    """Base class for time-based features."""
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Generate time-based feature for the given DatetimeIndex."""
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class WeekdayAndWeekend(TimeFeature):
    """Weekday vs weekend encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofweek + 1) // 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (pd.Index(index.isocalendar().week.astype('int64')) - 1) / 52.0 - 0.5

class SeasonOfYear(TimeFeature):
    """Season of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (((index.month // 3) + 3) % 4) / 3.0 - 0.5


class TimeFeatureGenerator:
    """Class to generate all time-related features from a given DatetimeIndex."""
    
    def __init__(self):
        # Define the list of all time-related feature classes to use
        self.time_features = [
            SecondOfMinute(),
            MinuteOfHour(),
            HourOfDay(),
            DayOfWeek(),
            WeekdayAndWeekend(),
            DayOfMonth(),
            DayOfYear(),
            MonthOfYear(),
            WeekOfYear(),
            SeasonOfYear()
        ]

    def generate_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate all time-related features for the given DatetimeIndex."""
        features = {}
        for feature in self.time_features:
            feature_name = feature.__class__.__name__
            features[feature_name] = feature(index)
        return pd.DataFrame(features, index=index)