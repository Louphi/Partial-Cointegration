"""Point-in-time constituent lookup. co_tic requires external historical mapping."""

import pandas as pd
from partial_cointegration.data import members_at


class DataCollection:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def list_nasdaq_constituents_by_year(self, year=2023):
        """Formation-end membership preceding the named trading year."""
        return members_at(
            self.data,
            pd.Timestamp(year=int(year), month=1, day=1) - pd.Timedelta(days=1),
        )

    def members_at(self, date):
        return members_at(self.data, date)

    def create_nasdaq_constituents_daily(self, *, end_date=None):
        if end_date is None:
            raise ValueError("Specify an explicit end_date for reproducibility")
        days = pd.date_range(pd.to_datetime(self.data["from"]).min(), end_date)
        return {str(day.date()): set(members_at(self.data, day)) for day in days}
