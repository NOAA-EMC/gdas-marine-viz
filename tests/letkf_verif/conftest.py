import os
import sys

# The application is a flat directory of scripts, as the others here are, and
# its modules import each other by bare name (`from lv_common import ...`).
# Importing one by dotted path alone therefore would not resolve its siblings,
# so put the application directory on sys.path the same way its own entry
# points do.
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'applications', 'letkf_verif')
if APP not in sys.path:
    sys.path.insert(0, APP)
