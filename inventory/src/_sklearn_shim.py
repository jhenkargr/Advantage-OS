"""
Shim to pre-load sklearn modules that have WDAC-blocked .pyd files.

Stubs out the C-extension symbols that the pure-Python import chain
expects so that LogisticRegression, MLPRegressor, StandardScaler, and
Pipeline can all be loaded without touching blocked .pyd files.

Usage:
    import _sklearn_shim   # must be the FIRST sklearn-related import
    # ... then import sklearn normally
"""

import sys
import types


def _make_fake(name, attrs=None):
    mod = types.ModuleType(name)
    for a, v in (attrs or {}).items():
        setattr(mod, a, v)
    sys.modules[name] = mod
    return mod


# The root blocker: _dist_metrics.pyd
_make_fake("sklearn.metrics._dist_metrics", {
    "BOOL_METRICS": {},
    "METRIC_MAPPING64": {},
    "DistanceMetric": type("DistanceMetric", (), {}),
    "DistanceMetric64": type("DistanceMetric64", (), {}),
    "DistanceMetric32": type("DistanceMetric32", (), {}),
})

# Downstream Cython modules that import _dist_metrics at init time
for mod_name in [
    "sklearn.metrics._pairwise_distances_reduction._base",
    "sklearn.metrics._pairwise_distances_reduction._argkmin",
    "sklearn.metrics._pairwise_distances_reduction._argkmin_classmode",
    "sklearn.metrics._pairwise_distances_reduction._radius_neighbors",
    "sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode",
]:
    _make_fake(mod_name, {
        "ArgKmin32": None, "ArgKmin64": None,
        "RadiusNeighbors32": None, "RadiusNeighbors64": None,
        "ArgKminClassMode32": None, "ArgKminClassMode64": None,
        "RadiusNeighborsClassMode32": None, "RadiusNeighborsClassMode64": None,
    })

# The dispatcher that ties them together
class _FakeMetric:
    @staticmethod
    def valid_metrics():
        return set()

_make_fake("sklearn.metrics._pairwise_distances_reduction._dispatcher", {
    "ArgKmin": _FakeMetric,
    "RadiusNeighbors": _FakeMetric,
    "sqeuclidean_row_norms": lambda *a, **k: None,
})

# The __init__ that aggregates them
_make_fake("sklearn.metrics._pairwise_distances_reduction", {
    "ArgKmin": _FakeMetric,
    "RadiusNeighbors": _FakeMetric,
    "sqeuclidean_row_norms": lambda *a, **k: None,
})

# Tree splitter (already known blocked)
for mod_name in [
    "sklearn.tree._splitter",
    "sklearn.tree._criterion",
    "sklearn.tree._tree",
    "sklearn.tree._utils",
]:
    _make_fake(mod_name, {})
