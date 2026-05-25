import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import re

################################################################################
# Typography (set once at import time)
################################################################################

mpl.rcParams["mathtext.fontset"] = "stix"

################################################################################
# Font resolution
################################################################################

_FONT_ALIASES = {
    "times": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "times new roman": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "helvetica": [
        "Helvetica",
        "Helvetica Neue",
        "Nimbus Sans",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "arial": [
        "Arial",
        "Liberation Sans",
        "Nimbus Sans",
        "DejaVu Sans",
    ],
    "georgia": [
        "Georgia",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "garamond": [
        "Garamond",
        "EB Garamond",
        "Liberation Serif",
        "DejaVu Serif",
    ],
}


def _resolve_font_family(font):
    """Expand a font name (or list) to a fallback chain filtered to fonts
    actually installed on this system.

    Returns None when input is None (caller should leave rcParams alone).

    Raises
    ------
    ValueError
        If none of the candidate fonts (after alias expansion) are installed.
    TypeError
        If `font` is not None, a string, or a list/tuple of strings.
    """
    if font is None:
        return None

    from matplotlib import font_manager

    # Build candidate list by expanding aliases
    if isinstance(font, str):
        key = font.lower().strip()
        candidates = _FONT_ALIASES.get(key, [font])
    elif isinstance(font, (list, tuple)):
        candidates = []
        for f in font:
            if isinstance(f, str):
                candidates.extend(_FONT_ALIASES.get(f.lower().strip(), [f]))
            else:
                candidates.append(f)
    else:
        raise TypeError(
            f"font must be a string, a list/tuple of strings, or None. "
            f"Got {type(font).__name__}."
        )

    # Filter to fonts actually installed; dedupe while preserving order
    installed, seen = [], set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            font_manager.findfont(name, fallback_to_default=False)
            installed.append(name)
        except Exception:
            continue

    if not installed:
        raise ValueError(
            f"None of the requested fonts are installed on this system: "
            f"{candidates!r}. Available aliases: {sorted(_FONT_ALIASES)}. "
            f"To see what's available: "
            f"`sorted({{f.name for f in matplotlib.font_manager.fontManager.ttflist}})`."
        )

    return installed


TARGET_CRS = "EPSG:32636"  # UTM 36N covers Ukraine

################################################################################
# Geometry cache (Ukraine boundaries + city markers are shared across calls)
################################################################################

_UKRAINE = None
_CITIES = None


def _load_geometry():
    """Load Ukraine admin-1 boundaries and major city markers once."""
    global _UKRAINE, _CITIES
    if _UKRAINE is None:
        _UKRAINE = gpd.read_file(
            "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_UKR_1.json"
        ).to_crs(TARGET_CRS)

        cities = pd.DataFrame(
            [
                ("Kyiv", 50.4501, 30.5234),
                ("Kharkiv", 49.9935, 36.2304),
                ("Donetsk", 48.0159, 37.8028),
                ("Luhansk", 48.5740, 39.3078),
                ("Mariupol", 47.0971, 37.5434),
                ("Zaporizhzhia", 47.8388, 35.1396),
                # Frontline corridor cities (Donetsk oblast)
                ("Avdiivka", 48.1394, 37.7479),
                ("Marinka", 47.9469, 37.5061),
                ("Pokrovsk", 48.2825, 37.1769),
            ],
            columns=["name", "lat", "lon"],
        )
        _CITIES = gpd.GeoDataFrame(
            cities,
            geometry=gpd.points_from_xy(cities["lon"], cities["lat"]),
            crs="EPSG:4326",
        ).to_crs(TARGET_CRS)
    return _UKRAINE, _CITIES


def _slugify(s):
    """'Air/drone strike' -> 'air_drone_strike' for use in filenames."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


################################################################################
# Main function
################################################################################


def plot_residual_map(
    df,
    group_value,
    *,
    group_col="sub_event_type",
    clip_to_oblasts=None,
    cities="auto",
    buffer_km=15,
    vmin=None,
    vmax=None,
    title=None,
    figsize=(7.5, 6),
    font="times new roman",
    title_fontsize=11.5,
    label_fontsize=8,
    map_fontsize=9,
    save_dir="../images/maps",
    save=True,
):
    """
    Plot a residual map for one sub-event type (or event group).

    Parameters
    ----------
    df : DataFrame
        Residual dataframe. Must contain columns: latitude, longitude,
        error, abs_error, and the column named by `group_col`.
    group_value : str
        The sub-event type (or event group) to filter on, e.g. "Air/drone strike".
    group_col : str, default "sub_event_type"
        Column to filter on. Use "event_group" if plotting the
        collapsed-majors-plus-Other version.
    clip_to_oblasts : list of str or None, default None
        Optional list of GADM oblast names (or case-insensitive substrings)
        to zoom the map to. Examples: ["Donets'k"], ["Donets'k", "Luhans'k"].
        If None, the full Ukraine extent is shown.
    cities : "auto", "national", "frontline", None, or list, default "auto"
        Which city labels to show on the map.

        - "auto"      : national cities only when Ukraine-wide; all cities
                        when zoomed via `clip_to_oblasts`.
        - "national"  : Kyiv, Kharkiv, Donetsk, Luhansk, Mariupol, Zaporizhzhia.
        - "frontline" : Donetsk, Avdiivka, Marinka, Pokrovsk, Luhansk, Mariupol.
        - None        : no city labels.
        - list        : explicit list of city names to show.
    buffer_km : float, default 15
        Padding (in kilometres) around the clipped region when
        `clip_to_oblasts` is set. Ignored when plotting the full Ukraine extent.
    vmin, vmax : float, optional
        Symmetric colour scale bounds. If not provided, computed as the
        99th-percentile-clipped symmetric range of |residual| within the
        selected subset. Pass these in explicitly if you want consistent
        scales across multiple calls (e.g., for a manual 2x2 layout).
    title : str, optional
        Figure title. Auto-generated from `group_value` if None.
    figsize : (float, float), default (7.5, 6)
        Figure size in inches.
    font : str, list, or None, default "times new roman"
        Font name, alias (e.g. "times", "helvetica", "georgia"), or list of
        fallback names for all text in the figure. Aliases are expanded
        automatically. Pass None to leave matplotlib's font settings unchanged.
        Raises ValueError if none of the candidates are installed.
    title_fontsize : float, default 11.5
        Font size for the figure title.
    label_fontsize : float, default 8
        Font size for city labels on the map.
    map_fontsize : float, default 9
        Font size for colorbar, size legend, and scale bar text.
    save_dir : str, default "../images/maps"
        Output directory. Created if it does not exist.
    save : bool, default True
        If True, writes PDF, PNG, and SVG files to `save_dir`.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    # Resolve font to an installed fallback chain
    resolved = _resolve_font_family(font)
    if resolved is not None:
        mpl.rcParams["font.family"] = resolved

    # Filter and project
    subset = df[df[group_col] == group_value].copy()
    if len(subset) == 0:
        raise ValueError(f"No rows in df where {group_col} == {group_value!r}")

    gdf = gpd.GeoDataFrame(
        subset,
        geometry=gpd.points_from_xy(subset["longitude"], subset["latitude"]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    # Color scale
    if vmax is None:
        vmax = float(np.ceil(np.percentile(gdf["error"].abs(), 99) / 5) * 5)
        vmax = max(vmax, 5)  # floor to avoid degenerate scales
    if vmin is None:
        vmin = -vmax

    # Load shared geometry
    ukraine, cities_gdf_full = _load_geometry()

    # Select which cities to show
    if cities == "auto":
        if clip_to_oblasts is None:
            keep = {
                "Kyiv",
                "Kharkiv",
                "Donetsk",
                "Luhansk",
                "Mariupol",
                "Zaporizhzhia",
            }
        else:
            keep = set(cities_gdf_full["name"])
    elif cities == "national":
        keep = {
            "Kyiv",
            "Kharkiv",
            "Donetsk",
            "Luhansk",
            "Mariupol",
            "Zaporizhzhia",
        }
    elif cities == "frontline":
        keep = {
            "Donetsk",
            "Avdiivka",
            "Marinka",
            "Pokrovsk",
            "Luhansk",
            "Mariupol",
        }
    elif cities is None:
        keep = set()
    elif isinstance(cities, (list, tuple, set)):
        keep = set(cities)
    else:
        raise ValueError(
            f"cities must be 'auto', 'national', 'frontline', None, or a list. "
            f"Got {cities!r}"
        )

    cities_gdf = cities_gdf_full[cities_gdf_full["name"].isin(keep)]

    # Figure
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Base oblast polygons
    ukraine.plot(
        ax=ax,
        color="#f6f4ef",
        edgecolor="#9e9e9e",
        linewidth=0.55,
    )

    # Residual scatter
    sc = ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf["error"],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        s=20 + gdf["abs_error"] * 6,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
    )

    # City markers
    ax.scatter(
        cities_gdf.geometry.x,
        cities_gdf.geometry.y,
        marker="s",
        s=14,
        color="black",
        zorder=10,
    )
    label_offsets = {
        "Mariupol": (5, -10),
        "Donetsk": (8, -10),  # below the marker; cluster is above and west
        "Luhansk": (6, 4),
        "Avdiivka": (6, 4),  # upper-right of marker
        "Marinka": (-15, -25),  # left of marker
        "Pokrovsk": (-40, 4),  # upper-left, in the emptier area NW of cluster
        "Zaporizhzhia": (-50, -8),
    }
    for _, row in cities_gdf.iterrows():
        dx, dy = label_offsets.get(row["name"], (5, 5))
        ax.annotate(
            row["name"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=label_fontsize,
            color="#222",
            zorder=11,
        )

    # Axis cosmetics
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    if title is None:
        title = (
            f"Spatial distribution of prediction residuals: "
            f"{group_value} (n = {len(gdf):,})"
        )
    ax.set_title(title, fontsize=title_fontsize, pad=8)

    # Colour bar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, aspect=22)
    cbar.set_label(
        "Residual (predicted − actual fatalities)",
        fontsize=map_fontsize,
    )
    cbar.ax.tick_params(labelsize=map_fontsize)

    # Size legend
    ref_sizes = [5, 10, 20]
    size_handles = [
        plt.scatter(
            [],
            [],
            s=20 + r * 6,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.3,
            label=str(r),
        )
        for r in ref_sizes
    ]
    legend = ax.legend(
        handles=size_handles,
        title="|residual|",
        loc="lower left",
        fontsize=map_fontsize,
        title_fontsize=map_fontsize,
        frameon=True,
        framealpha=0.9,
        edgecolor="#999",
        borderpad=0.6,
        labelspacing=0.9,
    )
    ax.add_artist(legend)

    # Optional zoom to one or more oblasts
    if clip_to_oblasts is not None:
        pattern = "|".join(clip_to_oblasts)
        clipped = ukraine[
            ukraine["NAME_1"].str.contains(
                pattern,
                case=False,
                na=False,
            )
        ]
        if len(clipped) == 0:
            available = sorted(ukraine["NAME_1"].unique())
            raise ValueError(
                f"No GADM oblasts matched {clip_to_oblasts}.\n"
                f"Available names: {available}"
            )
        minx, miny, maxx, maxy = clipped.total_bounds
        buf = buffer_km * 1000
        ax.set_xlim(minx - buf, maxx + buf)
        ax.set_ylim(miny - buf, maxy + buf)

    # Scale bar (50 km)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    bar_len_m = 50_000
    bx = xlim[1] - (xlim[1] - xlim[0]) * 0.22
    by = ylim[0] + (ylim[1] - ylim[0]) * 0.04
    ax.plot([bx, bx + bar_len_m], [by, by], color="black", lw=1.4)
    ax.text(
        bx + bar_len_m / 2,
        by + (ylim[1] - ylim[0]) * 0.012,
        "50 km",
        ha="center",
        va="bottom",
        fontsize=map_fontsize,
    )

    # Save
    if save:
        os.makedirs(save_dir, exist_ok=True)
        slug = _slugify(group_value)
        fig.savefig(
            f"{save_dir}/residual_map_{slug}.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            f"{save_dir}/residual_map_{slug}.png",
            dpi=400,
            bbox_inches="tight",
        )
        fig.savefig(
            f"{save_dir}/residual_map_{slug}.svg",
            bbox_inches="tight",
        )

    plt.show()
