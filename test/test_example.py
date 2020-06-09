import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / '../example'))

import example


def test_example():
    # In case something goes weird with the import and we import the folder of the same name one level above
    if hasattr(example, 'main'):
        example.main()
    elif hasattr(example, 'example'):
        example.example.main()
    else:
        raise RuntimeError("Can't find example.main")
