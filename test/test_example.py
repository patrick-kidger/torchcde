import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / '../example'))

import example
import logsignature_example
import irregular_data


def test_example():
    example.main(num_epochs=3)


def test_logsignature_example():
    logsignature_example.main(num_epochs=1)


def test_irregular_data():
    irregular_data.irregular_data()
