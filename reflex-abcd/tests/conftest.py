"""Put ``src/`` on the path so tests import ``reflex`` without an install step."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
