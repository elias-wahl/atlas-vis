import itertools
import functools


class Aliases:
    """
    Centralized registry for resolving heterogeneous atmospheric variable and platform names.

    Dynamically expands base aliases using common meteorological suffixes and casing
    to generate exhaustive exact-match lookup lists (>500 strings per key).
    Dictionaries are strictly ordered by key length (descending) to ensure that specific
    variables are matched before general ones during iterative lookups.
    """

    def __init__(self) -> None:
        """Initialize the expanded and sorted alias dictionaries."""
        self._input_dict = self._build_input_aliases()
        self._vars_dict = self._build_var_aliases()

    def for_input(self, key: str) -> list[str]:
        """
        Retrieve the list of complete string matches for a specific input platform.

        Args:
            key (str): The requested input type (e.g., 'uas', 'simulation').

        Returns:
            list[str]: A heavily expanded list of exact string permutations.

        Raises:
            KeyError: If the input platform is not defined in the base registry.
        """
        key = key.lower().strip()
        if key not in self._input_dict:
            raise KeyError(f"Input platform '{key}' not found in registry.")
        return self._input_dict[key]

    def for_vars(self, key: str) -> list[str]:
        """
        Retrieve the list of complete string matches for a MetPy atmospheric variable.

        Args:
            key (str): The requested MetPy variable name (e.g., 'temperature').

        Returns:
            list[str]: A heavily expanded list of exact string permutations.

        Raises:
            KeyError: If the atmospheric variable is not defined in the base registry.
        """
        key = key.lower().strip()
        if key not in self._vars_dict:
            raise KeyError(f"Atmospheric variable '{key}' not found in registry.")
        return self._vars_dict[key]

    def get_match(self, alias: str) -> str | None:
        """
        Retrieve the first matching parameter for a given alias.

        Leverages the underlying length-descending search order to avoid
        mismatches (e.g., catching 'potential_temperature' before 'temperature').

        Args:
            alias (str): The alias string to look up.

        Returns:
            str | None: The matched key, or None if no match is found.
        """
        matches = self._find_matches(alias)
        return matches[0] if matches else None

    def get_all_matches(self, alias: str) -> list[str]:
        """
        Retrieve all matching parameters for a given alias.

        Args:
            alias (str): The alias string to look up.

        Returns:
            list[str]: All keys that contain the given alias.
        """
        return self._find_matches(alias)

    @functools.lru_cache(maxsize=128)
    def get_fuzzy_match(self, alias: str) -> str | None:
        """
        Retrieve the first matching parameter using substring/fuzzy matching.
        This handles cases where the alias contains units or extra metadata.
        """
        clean_alias = alias.lower().strip()
        
        # Strip common formatting like text in parentheses
        import re
        stripped_alias = re.sub(r'\(.*?\)', '', clean_alias).strip()

        # Check for exact matches first
        exact_match = self.get_match(stripped_alias)
        if exact_match:
            return exact_match

        # Then substring matches
        # Because dicts are ordered by length descending, longer more specific variables match first
        for registry in (self._vars_dict, self._input_dict):
            for key, expanded_aliases in registry.items():
                for exp_alias in expanded_aliases:
                    if len(exp_alias) < 3:
                        # For very short aliases like 't', 'e', 'u', 'v', require exact word match
                        pattern = r'\b' + re.escape(exp_alias) + r'\b'
                        if re.search(pattern, stripped_alias):
                            return key
                    else:
                        if exp_alias in stripped_alias:
                            return key
        return None

    def _find_matches(self, alias: str) -> list[str]:
        """
        Base function to find all matching keys for a given alias.

        Searches through the ordered dictionaries to ensure longer, more
        specific keys are evaluated first.

        Args:
            alias (str): The alias string to look up.

        Returns:
            list[str]: A list of all matching dictionary keys.
        """
        clean_alias = alias.lower().strip()
        matches: list[str] = []

        # Iterate through both dictionaries. Since they were built with
        # length-descending sorted keys, specific keys are naturally checked first.
        for registry in (self._vars_dict, self._input_dict):
            for key, expanded_aliases in registry.items():
                if clean_alias in expanded_aliases:
                    matches.append(key)

        return matches

    def _expand_aliases(self, base_list: list[str], is_var: bool = True) -> list[str]:
        """
        Procedurally generate massive exact-match string arrays.

        Combines base names with expanded meteorological prefixes, suffixes, casing,
        and spacing variations. Deduplicates the final output using a set.

        Args:
            base_list (list[str]): The core root names for a variable or platform.
            is_var (bool): Whether to include variable-specific suffixes (e.g., '_sfc').

        Returns:
            list[str]: A sorted, deduplicated list of all string permutations.
        """
        expanded: set[str] = set()

        # Expanded suffix list to deeply multiply permutations
        suffixes = [
            "",
            "_mean",
            "_avg",
            "_obs",
            "_val",
            "_inst",
            "_raw",
            " mean",
            " avg",
            "_01",
            "_02",
            "_1",
            "_2",
            "_00",
            "_qc",
            "_qa",
            "_curr",
            "_final",
            "_1d",
            "_3d",
        ]
        if is_var:
            suffixes.extend(["_sfc", "_2m", "_10m", "_profile", "_lvl", "_surface", "_lev"])

        # Expanded prefix list
        prefixes = ["", "val_", "obs_", "avg_", "mean_", "raw_", "inst_", "curr_"]

        for base in base_list:
            for prefix, suffix in itertools.product(prefixes, suffixes):
                combined = f"{prefix}{base}{suffix}"

                # Casing variations
                expanded.add(combined.lower())
                expanded.add(combined.upper())
                expanded.add(combined.title())

                # Spacer variations
                expanded.add(combined.replace("_", " "))
                expanded.add(combined.replace(" ", "_"))
                expanded.add(combined.replace("_", "-"))
                expanded.add(combined.replace("-", "_"))

        return sorted(list(expanded))

    def _build_input_aliases(self) -> dict[str, list[str]]:
        """Construct the platform mapping, ordered by key length descending."""
        base_inputs = {
            "uas": [
                "uas",
                "uav",
                "drone",
                "rpas",
                "uncrewed_aerial_system",
                "copter",
                "multicopter",
                "fixed_wing",
                "quadcopter",
                "flight_vector",
            ],
            "aircraft": [
                "aircraft",
                "airplane",
                "flight_campaign",
                "plane",
                "research_aircraft",
                "halo",
                "falcon",
                "p3",
                "c130",
                "manned_aircraft",
            ],
            "lidar": [
                "lidar",
                "doppler_lidar",
                "wind_lidar",
                "ceilometer",
                "laser_radar",
                "profiler",
                "aerosol_lidar",
                "raman_lidar",
                "ceil",
            ],
            "station": [
                "station",
                "aws",
                "weather_station",
                "met_mast",
                "surface_station",
                "mesonet",
                "synop",
                "tower",
                "flux_tower",
                "buoy",
            ],
            "simulation": [
                "simulation",
                "wrf",
                "eulag",
                "model",
                "nwp",
                "icon",
                "ecmwf",
                "gfs",
                "arpege",
                "cosmo",
                "les",
                "dns",
                "output",
            ],
            "sounding": [
                "sounding",
                "radiosonde",
                "sonde",
                "balloon",
                "weather_balloon",
                "profile",
                "dropsonde",
                "tethered_balloon",
                "rawinsonde",
                "upper_air",
            ],
        }

        # Sort keys by length descending to ensure specific multi-word platforms are checked first
        sorted_keys = sorted(base_inputs.keys(), key=len, reverse=True)
        return {k: self._expand_aliases(base_inputs[k], is_var=False) for k in sorted_keys}

    def _build_var_aliases(self) -> dict[str, list[str]]:
        """Construct the variable mapping for the top 50 MetPy targets, ordered by key length descending."""
        base_vars = {
            # Time & Coordinates
            "time": [
                "time",
                "datum",
                "zeit",
                "date",
                "datetime",
                "timestamp",
                "Datum/Zeit",
                "valid_time",
                "time_obs",
            ],
            # Thermodynamics & Temperature
            "temperature": [
                "t",
                "temp",
                "temperature",
                "t_air",
                "tair",
                "t2m",
                "t_sfc",
                "air_temperature",
                "pt100",
                "lufttemperatur",
            ],
            "potential_temperature": [
                "theta",
                "th",
                "potential_temperature",
                "pot_temp",
                "pottemp",
                "thta",
                "air_potential_temperature",
                "t_pot",
            ],
            "equivalent_potential_temperature": [
                "theta_e",
                "thetae",
                "equiv_pot_temp",
                "equivalent_potential_temperature",
                "eth",
                "th_e",
            ],
            "virtual_temperature": [
                "t_v",
                "tv",
                "virtual_temperature",
                "virt_temp",
                "virtual_temp",
            ],
            "dewpoint": [
                "td",
                "dewpoint",
                "dew_point",
                "tdew",
                "dp",
                "dpt",
                "dewpoint_temperature",
            ],
            "skin_temperature": [
                "tskc",
                "tskin",
                "skin_temperature",
                "t_skin",
                "sfc_temp",
                "surface_temperature",
            ],
            # Moisture
            "relative_humidity": [
                "rh",
                "rel_hum",
                "relative_humidity",
                "humidity",
                "relhum",
                "rhum",
                "hur",
                "hum",
                "relative feuchte",
                "rel_feuchte",
            ],
            "specific_humidity": [
                "q",
                "spfh",
                "specific_humidity",
                "spec_hum",
                "sh",
                "hus",
            ],
            "mixing_ratio": [
                "w",
                "mix_rat",
                "mixing_ratio",
                "water_vapor_mixing_ratio",
                "qvapor",
                "qv",
                "m_hu",
            ],
            "precipitable_water": [
                "pw",
                "pwat",
                "precipitable_water",
                "tcwv",
                "total_column_water",
                "prw",
            ],
            "cloud_liquid_water": [
                "qcloud",
                "qc",
                "clw",
                "cloud_liquid_water",
                "cld_liq",
                "clwvi",
            ],
            "cloud_ice_water": [
                "qice",
                "qi",
                "ciw",
                "cloud_ice_water",
                "cld_ice",
                "clivi",
            ],
            "rain_mixing_ratio": ["qrain", "qr", "rain_mixing_ratio", "rain_mr"],
            "snow_mixing_ratio": ["qsnow", "qs", "snow_mixing_ratio", "snow_mr"],
            # Wind & Kinematics
            "u_wind": [
                "u",
                "u_wind",
                "u_comp",
                "u_velocity",
                "zonal_wind",
                "ua",
                "u_grd",
            ],
            "v_wind": [
                "v",
                "v_wind",
                "v_comp",
                "v_velocity",
                "meridional_wind",
                "va",
                "v_grd",
            ],
            "w_wind": [
                "w",
                "w_wind",
                "vertical_velocity",
                "w_velocity",
                "omega",
                "dz_dt",
                "wap",
            ],
            "wind_speed": [
                "ws",
                "wspd",
                "wind_speed",
                "speed",
                "wind_mag",
                "uv",
                "sfc_wind",
                "ff",
                "wspeed",
                "windgeschwindigkeit",
            ],
            "wind_gust": [
                "gust",
                "wind_gust",
                "windspitze",
            ],
            "wind_gust_direction": [
                "wind_gust_direction",
                "gust_dir",
                "windrichtung der windspitze",
            ],
            "wind_direction": [
                "wd",
                "wdir",
                "wind_direction",
                "wind_dir",
                "direction",
                "phi",
                "dd",
                "windrichtung",
            ],
            "vorticity": [
                "vort",
                "vorticity",
                "relative_vorticity",
                "abs_vort",
                "zeta",
                "vo",
            ],
            "divergence": ["div", "divergence", "wind_divergence", "divg", "d"],
            # Pressure
            "pressure": ["p", "pres", "pressure", "air_pressure", "p_sfc", "ps", "luftdruck"],
            "sea_level_pressure": [
                "slp",
                "mslp",
                "sea_level_pressure",
                "mean_sea_level_pressure",
                "prmsl",
                "psl",
            ],
            "surface_pressure": [
                "psfc",
                "surface_pressure",
                "sfc_pres",
                "sp",
                "ps",
            ],
            "geopotential": [
                "phi",
                "geopotential",
                "geopot",
                "zg_phi",
            ],
            "geopotential_height": [
                "z",
                "zg",
                "gph",
                "geopotential_height",
                "height",
                "gh",
            ],
            # Radiation
            "shortwave_radiation_down": [
                "swdown",
                "swd",
                "rsds",
                "shortwave_down",
                "downward_shortwave",
                "glw",
                "ssrd",
                "kw-strahlung unten",
            ],
            "shortwave_radiation_up": [
                "swup",
                "swu",
                "rsus",
                "shortwave_up",
                "upward_shortwave",
                "ssru",
                "kw-strahlung oben",
            ],
            "longwave_radiation_down": [
                "lwdown",
                "lwd",
                "rlds",
                "longwave_down",
                "downward_longwave",
                "strd",
                "lw-strahlung unten",
            ],
            "longwave_radiation_up": [
                "lwup",
                "lwu",
                "rlus",
                "longwave_up",
                "upward_longwave",
                "stru",
                "lw-strahlung oben",
            ],
            "net_radiation": ["rnet", "net_rad", "net_radiation", "rn"],
            "albedo": ["albedo", "alb", "surface_albedo", "sw_albedo", "al"],
            # Surface & Soil
            "soil_temperature": [
                "tso",
                "soil_temp",
                "soil_temperature",
                "tslb",
                "st",
                "tsl",
            ],
            "soil_moisture": [
                "smois",
                "soil_moisture",
                "soil_water",
                "sm",
                "swc",
                "mrsos",
            ],
            "sensible_heat_flux": ["hfx", "shf", "sensible_heat", "sshf", "hfss"],
            "latent_heat_flux": ["lh", "lhf", "latent_heat", "slhf", "hfls"],
            "ground_heat_flux": ["grdflx", "ghf", "ground_heat", "g", "hfg"],
            "surface_roughness": [
                "z0",
                "roughness",
                "surface_roughness",
                "sfcrgh",
                "zr",
            ],
            # Precipitation
            "precipitation_rate": [
                "pr",
                "precip_rate",
                "rain_rate",
                "precipitation_flux",
                "prate",
            ],
            "total_precipitation": [
                "precip",
                "tp",
                "total_precip",
                "rain",
                "acc_precip",
                "prcp",
            ],
            "convective_precipitation": [
                "rainc",
                "cp",
                "convective_precip",
                "c_precip",
            ],
            "large_scale_precipitation": [
                "rainnc",
                "lsp",
                "large_scale_precip",
                "ls_precip",
            ],
            "snow_depth": ["snowh", "snwd", "snow_depth", "snd", "sd"],
            "snow_water_equivalent": ["swe", "snow", "snow_water_equivalent", "snw"],
            # Boundary Layer & Turbulence
            "planetary_boundary_layer_height": [
                "pblh",
                "hpbl",
                "pbl_height",
                "boundary_layer_height",
                "zi",
            ],
            "turbulent_kinetic_energy": ["tke", "e", "turbulent_kinetic_energy", "k"],
            "friction_velocity": ["ust", "ustar", "friction_velocity", "u_star"],
            "richardson_number": ["ri", "richardson", "bulk_richardson", "brn"],
            # Other Diagnostics
            "visibility": [
                "vis",
                "visibility",
                "vis_dist",
                "meteorological_optical_range",
            ],
            "cloud_fraction": ["cloud_frac", "cldfra", "cf", "cloud_cover", "clt"],
            "convective_available_potential_energy": [
                "cape",
                "convective_available_potential_energy",
            ],
            "convective_inhibition": ["cin", "cins", "convective_inhibition"],
            "lifted_index": ["li", "lifted_index", "pli"],
        }

        # Sort keys by length descending to ensure specific multi-word variables are checked first
        sorted_keys = sorted(base_vars.keys(), key=len, reverse=True)
        return {k: self._expand_aliases(base_vars[k], is_var=True) for k in sorted_keys}

    @property
    def ordered_var_keys(self) -> list[str]:
        """Return the pre-sorted list of variable keys for iterative matching loops."""
        return list(self._vars_dict.keys())


aliases = Aliases()
