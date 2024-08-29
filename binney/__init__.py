from .binney import *
from .binney import __all__
from .binney import __doc__

from .binney import cli as cli

from pathlib import Path

try:
    import polars as pl
    class BinDirectoryDF(BinDirectory):
        def read_all(self, overwrite=False) -> pl.LazyFrame:
            """
                Retrieve a lazy polars dataframe for processing all of the photons in this bin directory
            """
            files = self.convert_all(overwrite=overwrite)
            return pl.scan_parquet(files)

        def read_timerange(self, timerange: TimestampRange, overwrite=False) -> pl.LazyFrame:
            """
                Retrieve a lazy polars dataframe for processing a specific timerange of photons in
                this bin directory
            """
            files = self.convert_timerange(timerange, overwrite=overwrite)
            return pl.scan_parquet(files)

        def read_timeranges(self, timeranges: list[TimestampRange], overwrite=False) -> pl.LazyFrame:
            """
                Retrieve a lazy polars dataframe for processing each timerange in timeranges
            """
            fileses = self.convert_timeranges(timeranges, overwrite=overwrite)
            return [pl.scan_parquet(files) for files in fileses]

    __all__ = [BinDirectoryDF] + __all__
except ImportError:
    pass
