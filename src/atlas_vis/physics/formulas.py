from typing import Any

import metpy.calc as mpcalc

# Level 3 Fallback: General MetPy equations registered as purely functional lambdas.
# All internal variables are strictly pre-quantified during the parsing phase.
# Wind fields are standardized directly to 'u' and 'v'.
GENERAL_EQUATIONS: dict[str, dict[str, Any]] = {
    "potential_temperature": {
        "dependencies": ["temperature", "pressure"],
        "func": lambda ds, t, p: mpcalc.potential_temperature(ds[p], ds[t]),
    },
    "equivalent_potential_temperature": {
        "dependencies": ["temperature", "pressure", "dewpoint"],
        "func": lambda ds, t, p, td: mpcalc.equivalent_potential_temperature(
            ds[p], ds[t], ds[td]
        ),
    },
    "wind_speed": {
        "dependencies": ["u", "v"],
        "func": lambda ds, u, v: mpcalc.wind_speed(ds[u], ds[v]),
    },
}
