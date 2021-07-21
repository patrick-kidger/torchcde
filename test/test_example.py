import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / '../example'))

import irregular_data  # noqa: E402
import logsignature_example  # noqa: E402
import time_series_classification  # noqa: E402

import markers  # noqa: E402


def test_irregular_data():
    irregular_data.irregular_data()


def test_time_series_classification():
    time_series_classification.main(num_epochs=3)


@markers.uses_signatory
def test_logsignature_example():
    logsignature_example.main(num_epochs=1)
