"""Site32 command-center application modules.

The legacy ``app.py`` remains the WSGI entry point while new functionality is
implemented in bounded modules and registered explicitly.
"""

from .site32_blueprint import register_site32
from .runtime import RuntimeController

__all__ = ["RuntimeController", "register_site32"]
