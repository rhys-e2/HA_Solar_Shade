"""A manual LAZ/LAS tile must line up with the map even when it is not
centred on the configured home location.

Regression test: the grid used to be centred on the file's centroid but
anchored (for overlay bounds, satellite request and lat/lng -> pixel mapping)
at the configured home location, shifting the height map by the distance
between the two.
"""

import math

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

from custom_components.solar_shade.geo import epsg_to_latlon, latlon_to_epsg
from custom_components.solar_shade.shadow_engine import (
    _centroid_latlon,
    process_lidar_file,
)
from custom_components.solar_shade.websocket_api import (
    _latlng_to_pixel,
    _site_bounds_latlng,
)

EPSG = 7855  # GDA2020 / MGA zone 55 (Tasmania)
HOME = (-42.88, 147.33)
BUILDING_H, GROUND_H = 25.0, 10.0


def _write_tile(path, offset_e, offset_n, size=300):
    """Flat ground plus a 12 m building at a known position.

    The tile spans home + [offset .. offset+size], so its centroid is
    (offset + size/2) metres from home in each axis.
    """
    he, hn = latlon_to_epsg(*HOME, EPSG)
    x0, y0 = he + offset_e, hn + offset_n
    gx, gy = np.meshgrid(np.arange(x0, x0 + size, 1.0), np.arange(y0, y0 + size, 1.0))
    # building near the middle of the tile
    be, bn = x0 + size * 0.6, y0 + size * 0.4
    bx, by = np.meshgrid(np.arange(be - 6, be + 6, 0.25), np.arange(bn - 6, bn + 6, 0.25))
    xs = np.r_[gx.ravel(), bx.ravel()]
    ys = np.r_[gy.ravel(), by.ravel()]
    zs = np.r_[np.full(gx.size, GROUND_H), np.full(bx.size, BUILDING_H)]
    cls = np.r_[np.full(gx.size, 2), np.full(bx.size, 6)].astype(np.uint8)

    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales = [0.01] * 3
    hdr.offsets = [x0, y0, 0]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z, las.classification = xs, ys, zs, cls
    las.write(str(path))
    return be, bn


def _metres_between(a, b):
    return (
        (b[0] - a[0]) * 111_320,
        (b[1] - a[1]) * 111_320 * math.cos(math.radians(a[0])),
    )


def _drawn_offset_m(site, true_en):
    """Distance between where the overlay draws the building and where it is."""
    rows, cols = site.dsm.shape
    rr, cc = np.where(site.dsm > (GROUND_H + BUILDING_H) / 2)
    assert len(rr), "building missing from DSM"
    b = _site_bounds_latlng(site)
    lat = b["north"] - (rr.mean() + 0.5) / rows * (b["north"] - b["south"])
    lon = b["west"] + (cc.mean() + 0.5) / cols * (b["east"] - b["west"])
    true_ll = epsg_to_latlon(*true_en, EPSG)
    dn, de = _metres_between(true_ll, (lat, lon))
    return math.hypot(dn, de), true_ll


@pytest.mark.parametrize(
    "offset_e,offset_n",
    [(-150, -150), (-100, -100), (200, -300), (-400, 250)],
    ids=["centred-ish", "200m-off", "mixed-signs", "far-ish"],
)
def test_overlay_and_zone_sampling_match_the_real_world(tmp_path, offset_e, offset_n):
    path = tmp_path / "tile.las"
    true_en = _write_tile(path, offset_e, offset_n)
    site = process_lidar_file(
        str(path), HOME[0], HOME[1], min_cell_size=1.0, epsg_override=EPSG
    )

    off_m, true_ll = _drawn_offset_m(site, true_en)
    assert off_m < 3.0, f"height map drawn {off_m:.0f} m from its true position"

    # A zone drawn on the building's real position must sample the building.
    row, col = _latlng_to_pixel(site, *true_ll)
    assert site.dsm[row, col] > (GROUND_H + BUILDING_H) / 2


def test_no_crs_in_file_falls_back_to_utm_and_still_lines_up(tmp_path):
    """No EPSG override and no CRS in the header: raw coordinates are treated as
    WGS84 UTM, which is within a couple of metres of MGA for this area."""
    path = tmp_path / "tile.las"
    true_en = _write_tile(path, -100, -100)
    site = process_lidar_file(str(path), HOME[0], HOME[1], min_cell_size=1.0)
    assert site.native_epsg == 0
    off_m, _ = _drawn_offset_m(site, true_en)
    assert off_m < 6.0


def test_model_is_anchored_at_file_centroid_not_home(tmp_path):
    path = tmp_path / "tile.las"
    _write_tile(path, 200, 200)  # centroid 350 m E and N of home
    site = process_lidar_file(
        str(path), HOME[0], HOME[1], min_cell_size=1.0, epsg_override=EPSG
    )
    dn, de = _metres_between(HOME, (site.latitude, site.longitude))
    assert dn == pytest.approx(350, abs=3)
    assert de == pytest.approx(350, abs=3)


class TestCentroidLatLon:
    def test_projects_the_centroid(self):
        he, hn = latlon_to_epsg(*HOME, EPSG)
        lat, lon = _centroid_latlon(he + 100, hn + 100, EPSG, *HOME)
        dn, de = _metres_between(HOME, (lat, lon))
        assert (dn, de) == (pytest.approx(100, abs=1), pytest.approx(100, abs=1))

    def test_implausibly_far_centroid_keeps_reference(self):
        """A wrong CRS must not corrupt the sun-position inputs."""
        # Same raw numbers read as UTM zone 55 are fine; read in a CRS for a
        # different continent they land thousands of km away.
        he, hn = latlon_to_epsg(*HOME, EPSG)
        assert _centroid_latlon(he, hn, 32618, *HOME) == HOME  # UTM 18N (US east coast)

    def test_unprojectable_centroid_keeps_reference(self):
        assert _centroid_latlon(1e15, 1e15, EPSG, *HOME) == HOME

    def test_unknown_epsg_keeps_reference(self):
        assert _centroid_latlon(500000, 5000000, 999999, *HOME) == HOME
