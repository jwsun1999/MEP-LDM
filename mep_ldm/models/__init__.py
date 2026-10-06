__all__ = [
    "get_model",
    "get_autoencoder",
    "PhysPropSurrogate",
    "SurrogateConfig",
]


def __getattr__(name):
    if name in {"get_model", "get_autoencoder"}:
        from .get_model import get_autoencoder, get_model

        return {"get_model": get_model, "get_autoencoder": get_autoencoder}[name]

    if name in {"PhysPropSurrogate", "SurrogateConfig"}:
        from .ppp import PhysPropSurrogate, SurrogateConfig

        return {
            "PhysPropSurrogate": PhysPropSurrogate,
            "SurrogateConfig": SurrogateConfig,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
