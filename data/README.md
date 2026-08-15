# Geospatial Data Sources

Place terrain and elevation files here for use with the `terrain_real` scenario.

## Directory Structure

```
data/
├── elevation/     # DEM GeoTIFF files (.tif)
├── osm_cache/     # Auto-populated by osmnx
└── README.md      # This file
```

## Obtaining Free Elevation Data

### SRTM DEM (30m resolution, global coverage)
- **Source**: USGS EarthExplorer — https://earthexplorer.usgs.gov
- **Alternative**: OpenTopography — https://opentopography.org
- **Format**: GeoTIFF (.tif), WGS84
- **Coverage**: 60°N to 56°S latitude

### ASTER GDEM v3 (30m resolution, global)
- **Source**: NASA Earthdata — https://search.earthdata.nasa.gov
- **Requires**: Free NASA Earthdata account

### Copernicus DEM (30m or 90m, global)
- **Source**: https://spacedata.copernicus.eu
- **Best option** for newest data with fewest artifacts

## Usage

```bash
# Download SRTM tile, place in data/elevation/
# Then run with real terrain:

convoy-commander run --scenario terrain_real \
  --elevation data/elevation/N32W085.tif \
  --osm-source "Fort Benning, Georgia" \
  --seed 42 --vehicles 8

# Or with bounding box (north,south,east,west):
convoy-commander run --scenario terrain_real \
  --elevation data/elevation/N32W085.tif \
  --geo-bounds 32.38,32.34,-84.88,-84.93 \
  --seed 42
```

## OpenStreetMap Roads

OSM road networks are downloaded automatically by `osmnx` when you provide
`--osm-source` (a place name) or `--geo-bounds`.  Downloaded data is cached
in `data/osm_cache/` for subsequent runs.

No manual download is needed for OSM data.
