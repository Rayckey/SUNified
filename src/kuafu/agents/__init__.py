"""Kuafu tools for agents that propose assets and SUN Programs."""

__all__ = ["ToolService"]


def __getattr__(name):
    if name == "ToolService":
        from .service import ToolService

        return ToolService
    raise AttributeError(name)
