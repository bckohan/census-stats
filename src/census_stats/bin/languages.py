"""
Count the languages spoken at home in an LA neighborhood council district.

Pipeline:
  1. Read the neighborhood council boundary from the certified NC shapefile.
  2. Find the 2020-vintage census tracts that overlap the boundary (TIGERweb).
  3. Aggregate ACS 5-year table C16001 ("Language Spoken at Home for the
     Population 5 Years and Over") across those tracts, using the keyless
     table-based summary files on www2.census.gov.

C16001 is the most detailed language table the Census Bureau publishes at
tract level; it collapses all languages into 12 groups plus English. The
detailed 39-language table (B16001) is only published for geographies of
100k+ people, so the honest neighborhood-level answer is a lower bound.
"""

import argparse
import csv
import math
import sys
import urllib.parse
from pathlib import Path

import geopandas as gpd
import requests

TIGERWEB_TRACTS = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "Tracts_Blocks/MapServer/0/query"
)
ACS_GROUP_METADATA = "https://api.census.gov/data/{year}/acs/acs5/groups/{table}.json"
ACS_SUMMARY_FILE = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/{year}/"
    "table-based-SF/data/5YRData/acsdt5y{year}-{table}.dat"
)
TABLE = "C16001"
# Equal-area CRS for California, used for overlap fractions.
AREA_CRS = "EPSG:3310"


def load_council_boundary(shapefile: Path, council: str):
    gdf = gpd.read_file(f"zip://{shapefile}" if shapefile.suffix == ".zip" else shapefile)
    match = gdf[gdf["NAME"].str.upper() == council.upper()]
    if match.empty:
        names = "\n  ".join(sorted(gdf["NAME"]))
        sys.exit(f"No council named {council!r} in {shapefile}. Available:\n  {names}")
    return match.to_crs(AREA_CRS).geometry.iloc[0]


def find_tracts(boundary, min_overlap: float):
    """Return {geoid: overlap_fraction} for tracts overlapping the boundary."""
    bounds = gpd.GeoSeries([boundary], crs=AREA_CRS).to_crs("EPSG:4326").total_bounds
    resp = requests.post(
        TIGERWEB_TRACTS,
        data={
            "geometry": ",".join(str(round(b, 6)) for b in bounds),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "GEOID,NAME",
            "returnGeometry": "true",
            "f": "geojson",
        },
        timeout=60,
    )
    resp.raise_for_status()
    tracts = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    tracts = tracts.to_crs(AREA_CRS)
    tracts["overlap"] = tracts.geometry.intersection(boundary).area / tracts.geometry.area
    tracts = tracts[tracts["overlap"] >= min_overlap]
    return dict(zip(tracts["GEOID"], tracts["overlap"]))


def language_variables(year: int):
    """Map summary-file column name -> language label for top-level rows.

    Uses the (keyless) ACS metadata endpoint. Labels look like
    'Estimate!!Total:!!Spanish:' — we keep only depth-2 rows, which are the
    per-language totals, skipping the English-ability breakdowns beneath them.
    """
    resp = requests.get(
        ACS_GROUP_METADATA.format(year=year, table=TABLE), timeout=60
    )
    resp.raise_for_status()
    variables = {}
    for name, info in resp.json()["variables"].items():
        if not name.endswith("E"):
            continue
        parts = [p.rstrip(":") for p in info["label"].split("!!")]
        # Keep 'Estimate!!Total:' and 'Estimate!!Total:!!<language group>:',
        # skipping the English-ability rows nested one level deeper.
        if len(parts) in (2, 3):
            # API name C16001_003E -> summary-file column C16001_E003
            column = f"{TABLE}_E{name[len(TABLE) + 1:-1]}"
            variables[column] = parts[-1]
    return variables


def fetch_table(year: int, cache_dir: Path) -> Path:
    """Download the national C16001 summary file (~70 MB) once and cache it."""
    url = ACS_SUMMARY_FILE.format(year=year, table=TABLE.lower())
    cached = cache_dir / Path(urllib.parse.urlparse(url).path).name
    if not cached.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...", file=sys.stderr)
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            tmp = cached.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(cached)
    return cached


def aggregate(table_file: Path, geoids: set[str], columns: dict[str, str]):
    """Sum estimates (and combine MOEs) for the given tracts.

    Returns {language: (estimate, moe)}. MOEs combine as sqrt(sum of squares),
    the standard ACS approximation for sums.
    """
    wanted = {f"1400000US{g}" for g in geoids}
    sums = {label: 0 for label in columns.values()}
    moe_sq = {label: 0.0 for label in columns.values()}
    found = set()
    with open(table_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            if row["GEO_ID"] not in wanted:
                continue
            found.add(row["GEO_ID"])
            for column, label in columns.items():
                sums[label] += int(row[column])
                moe = float(row[column.replace("_E", "_M", 1)])
                if moe >= 0:  # negative values are jam codes (e.g. -555 controlled)
                    moe_sq[label] += moe**2
            if found == wanted:
                break
    missing = wanted - found
    if missing:
        sys.exit(f"Tracts not found in summary file: {sorted(missing)}")
    return {label: (sums[label], math.sqrt(moe_sq[label])) for label in sums}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--council",
        default="ELYSIAN VALLEY RIVERSIDE NC",
        help="Neighborhood council NAME in the shapefile (default: Frogtown's EVRNC)",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=Path(__file__).parents[3] / "data" / "Neighborhood_Councils_Certified.zip",
        help="Path to the LA neighborhood council boundaries shapefile (.zip ok)",
    )
    parser.add_argument(
        "--year", type=int, default=2023, help="ACS 5-year vintage (default: 2023)"
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.10,
        help="Include a tract if at least this fraction of its area is inside "
        "the council boundary (default: 0.10)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).parents[3] / "data" / "cache",
        help="Directory for the cached ACS summary file download",
    )
    args = parser.parse_args(argv)

    boundary = load_council_boundary(args.shapefile, args.council)
    tracts = find_tracts(boundary, args.min_overlap)
    if not tracts:
        sys.exit("No census tracts met the overlap threshold.")

    columns = language_variables(args.year)
    table_file = fetch_table(args.year, args.cache_dir)
    stats = aggregate(table_file, set(tracts), columns)

    total, total_moe = stats.pop("Total")
    english_only = stats.pop("Speak only English")
    languages = sorted(stats.items(), key=lambda kv: -kv[1][0])
    spoken = [(label, est, moe) for label, (est, moe) in languages if est > 0]

    print(f"{args.council}  (ACS {args.year - 4}-{args.year} 5-year, table {TABLE})")
    print(f"\nCensus tracts used (>= {args.min_overlap:.0%} of tract area inside boundary):")
    for geoid, overlap in sorted(tracts.items()):
        print(f"  {geoid}  ({overlap:.0%} inside)")

    print(f"\nPopulation 5 years and over: {total:,} (+/- {total_moe:,.0f})")
    print(f"Speak only English at home:  {english_only[0]:,} (+/- {english_only[1]:,.0f})")
    print("\nLanguage groups spoken at home:")
    width = max(len(label) for label, _, _ in spoken) if spoken else 0
    for label, est, moe in spoken:
        print(f"  {label:<{width}}  {est:>6,} (+/- {moe:,.0f})")
    for label, (est, moe) in languages:
        if est == 0:
            print(f"  {label:<{width}}  {'none':>6} (+/- {moe:,.0f})")

    print(
        f"\n=> Speakers of at least {len(spoken) + 1} languages: English plus "
        f"{len(spoken)} of the census's 12 non-English language groups."
    )
    print(
        "   (C16001 is the most detailed language table published at tract level;\n"
        "   each group bundles many individual languages, so the true count is higher.)"
    )


if __name__ == "__main__":
    main()
