"""SDS-UFD-JSCC research prototype."""

__all__ = ["SIFDJSCC"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "SIFDJSCC":
        from .models.system import SIFDJSCC

        return SIFDJSCC
    raise AttributeError(name)
