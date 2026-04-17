_LAZY = {
    "BLEUScore": ("bleu", "BLEUScore"),
    "CHRFScore": ("bleu", "CHRFScore"),
    "TERScore":  ("bleu", "TERScore"),
    "BERTScore": ("bertscore", "BERTScore"),
    "SentenceBERTScore": ("sentbert", "SentenceBERTScore"),
    "COMETScore": ("comet", "COMETScore"),
    "MetricXScore": ("metricx", "MetricXScore"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, cls_name = _LAZY[name]
        mod = __import__(f"lightyear.metrics.{mod_name}", fromlist=[cls_name])
        return getattr(mod, cls_name)
    raise AttributeError(f"module 'lightyear.metrics' has no attribute {name!r}")


__all__ = list(_LAZY.keys())
