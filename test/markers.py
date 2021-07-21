import pytest
import sys


uses_signatory = pytest.mark.skipif(sys.platform == "darwin", reason="Signatory does not support MacOS")
