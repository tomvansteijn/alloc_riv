#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Tom van Steijn

import xarray as xr
import flopy
import imod

from scipy.interpolate import interp1d
import geopandas as gpd
import pandas as pd
import numpy as np


from pathlib import Path
import argparse
import logging
import yaml
import os

log = logging.getLogger(os.path.basename(__file__))


def get_parser():
    """get argumentparser and add arguments"""
    parser = argparse.ArgumentParser("description",)

    # Command line arguments
    parser.add_argument(
        "command",
        type=str,
        choices=["intersect", "allocate", "export"],
        help=("run command"),
    )
    parser.add_argument(
        "inputfile", type=str, help=("YAML input file containing keyword arguments")
    )
    parser.add_argument(
        "--debug", action="store_true", help=("set logging level to debug")
    )
    return parser


class ValueField(object):
    def __init__(self, name, nodata=None, valid_range=None, clip_range=None):
        self.name = name
        self.nodata = nodata
        self.valid_range = valid_range
        self.clip_range = clip_range


class Parameter(object):
    def __init__(self, name, filename, idfield, valuefields, buffer=0.):
        self.name = name
        self.filename = Path(filename)
        self.idfield = idfield
        self.valuefields = [ValueField(**f) for f in valuefields]
        self.buffer = buffer


def intersect(**kwargs):
    domainfile = Path(kwargs["domainfile"])
    riversfile = Path(kwargs["riversfile"])
    rivers_id = kwargs["rivers_id"]    
    cellsfile = Path(kwargs["cellsfile"])

    # create output dir
    cellsfile.parent.mkdir(exist_ok=True)

    # read domain
    log.info("reading domain file")
    domain = imod.idf.open(domainfile)
    dx, xmin, xmax = imod.util.coord_reference(domain.x)
    dy, ymin, ymax = imod.util.coord_reference(domain.y)
    bounds = xmin, ymin, xmax, ymax
    dy = -dy
    nrow, ncol = domain.shape[-2:]    

    # read rivers
    log.info("reading rivers file")
    rivers = gpd.read_file(riversfile, bbox=bounds)

    # create structured grid based on domain
    delr = np.ones(ncol) * dx  # confusing but correct
    delc = np.ones(nrow) * dy
    sgr = flopy.discretization.StructuredGrid(
        delc=delc,
        delr=delr,
        xoff=xmin,
        yoff=ymin,
        nrow=nrow,
        ncol=ncol,
        idomain=domain.values.squeeze(),
    )

    # create intersection object
    log.info("creating grid intersect")
    ixs = flopy.utils.gridintersect.GridIntersect(sgr)

    # intersect river with grid
    cells = []
    for i, (river_id, river) in enumerate(rivers.groupby(rivers_id)):
        log.debug(f"processing river {i:d}, id {river_id}")
        river_line = river.cascaded_union

        # intersect
        river_ix = ixs.intersect(river_line)
        if len(river_ix) == 0:
            log.warning(f"river {river_id} does not intersect with grid")
            continue

        # intersection to dataframe
        river_cells = pd.DataFrame(river_ix)
        river_cells.loc[:, "river_id"] = river_id
        river_cells = river_cells.set_index("river_id", append=True).reorder_levels(
            [1, 0]
        )
        river_cells.index.names = "river_id", "row_id"
        cells.append(river_cells)

    log.info("concatenating")
    cells = pd.concat(cells, axis=0)

    # get cell coordinates & geometry
    cells.loc[:, "row"] = cells.loc[:, "cellids"].map(lambda t: t[0])
    cells.loc[:, "col"] = cells.loc[:, "cellids"].map(lambda t: t[1])
    cells.loc[:, "x"] = sgr.xcellcenters[0, cells.loc[:, "col"]]
    cells.loc[:, "y"] = sgr.ycellcenters[cells.loc[:, "row"], 0]

    cells = gpd.GeoDataFrame(
        cells, geometry=gpd.points_from_xy(cells.loc[:, "x"], cells.loc[:, "y"])
    )

    # write to disk
    log.info("writing river cells to disk")
    cells.to_pickle(cellsfile)


def allocate(**kwargs):
    domainfile = Path(kwargs["domainfile"])
    riversfile = Path(kwargs["riversfile"])
    rivers_id = kwargs["rivers_id"]
    parameters = [Parameter(**p) for p in kwargs["parameters"]]
    cellsfile = Path(kwargs["cellsfile"])
    allocatedfile = Path(kwargs["allocatedfile"])

    # create output dir
    allocatedfile.parent.mkdir(exist_ok=True)

    # read domain
    log.info("reading domain file")
    domain = imod.idf.open(domainfile)
    dx, xmin, xmax = imod.util.coord_reference(domain.x)
    dy, ymin, ymax = imod.util.coord_reference(domain.y)
    bounds = xmin, ymin, xmax, ymax

    # read rivers
    log.info("reading rivers file")
    rivers = gpd.read_file(riversfile, bbox=bounds)

    # read cells
    log.info("reading cells from file")
    cells = pd.read_pickle(cellsfile)

    # read points
    for parameter in parameters:
        log.info(f"processing {parameter.name:}")
        bbox = (bounds[0] - parameter.buffer,
            bounds[1] - parameter.buffer,
            bounds[2] + parameter.buffer,
            bounds[3] + parameter.buffer,
            )
        points = gpd.read_file(parameter.filename, bbox=bbox)

        # replace nodata and values outside valid range with nan
        for valuefield in parameter.valuefields:
            if valuefield.nodata is not None:
                is_nodata = points.loc[:, valuefield.name].isin(valuefield.nodata)
                points.loc[is_nodata, valuefield.name] = np.nan

            if valuefield.valid_range is not None:
                in_valid_range = points.loc[:, valuefield.name].between(
                    *valuefield.valid_range, inclusive=False,
                )
                points.loc[~in_valid_range, valuefield.name] = np.nan

            if valuefield.clip_range is not None:
                points.loc[:, valuefield.name] = points.loc[:, valuefield.name].clip(
                    **valuefield.clip_range,
                )

        # loop over rivers and interpolate
        points_id = parameter.idfield
        points_values = [vf.name for vf in parameter.valuefields]
        allocated = []
        for i, (river_id, river) in enumerate(rivers.groupby(rivers_id)):
            log.debug(f"processing river {i:d}, id {river_id}")
            river_line = river.cascaded_union

            # select interpolation points using id
            is_river_point = points.loc[:, points_id] == river_id
            if not is_river_point.sum() > 2:
                continue
            river_points = points.loc[is_river_point, :].copy()

            # distance of points along river line
            river_points.loc[:, "distance"] = [
                river_line.project(p) for p in river_points.geometry
            ]

            # create 1d interpolator
            f = interp1d(
                river_points.loc[:, "distance"],
                river_points.loc[:, points_values],
                axis=0,
                fill_value="extrapolate",
            )

            # retrieve cells
            try:
                river_cells = cells.loc[river_id, :].copy()
            except KeyError:
                continue

            # distance of cell centers along river line
            river_cells.loc[:, "distance"] = [
                river_line.project(p) for p in river_cells.geometry
            ]

            # interpolate
            river_interp = f(river_cells.loc[:, "distance"])

            # interpolation to dataframe
            cell_values = pd.DataFrame(river_interp, columns=points_values)
            river_cells = pd.concat([river_cells, cell_values], axis=1)
            river_cells = river_cells.groupby(level=0, axis=1).last()
            river_cells.loc[:, "river_id"] = river_id
            river_cells = river_cells.set_index("river_id", append=True).reorder_levels(
                [1, 0]
            )
            river_cells.index.names = "river_id", "row_id"
            allocated.append(river_cells)

        log.info("concatenating")
        cells = pd.concat(allocated, axis=0)

    log.info("writing allocated river cells to disk")
    cells.to_pickle(allocatedfile)


def export(**kwargs):
    domainfile = Path(kwargs["domainfile"])
    allocatedfile = Path(kwargs["allocatedfile"])
    fieldnames = kwargs["fieldnames"]
    stages = kwargs["stages"]
    resultdir = Path(kwargs["resultdir"])

    resultdir.mkdir(exist_ok=True)

    log.info("reading domain file")
    domain = imod.idf.open(domainfile)

    log.info("reading allocated cells from file")
    cells = pd.read_pickle(allocatedfile)
    cells = cells.rename(columns={v: k for k, v in fieldnames.items()})

    log.info("calculating wet area")
    cells.loc[:, "wet_area"] = (
        cells.loc[:, "river_width"] * cells.loc[:, "lengths"]
    ).fillna(0.0)
    log.info("calculating conductance")
    cells.loc[:, "conductance"] = (
        cells.loc[:, "wet_area"] / cells.loc[:, "drainage_resistance"]
    ).fillna(0.0)
    log.info("calculating conductance fraction")
    cells.loc[:, "conductance_fraction"] = cells.groupby(["row", "col"])[
        "conductance"
    ].transform(lambda s: s.div(s.sum()))
    log.info("calculating infiltration factor")
    cells.loc[:, "infiltration_factor"] = (
        cells.loc[:, "drainage_resistance"] / cells.loc[:, "infiltration_resistance"]
    ).fillna(0.0)
    log.info("sum by row, col")
    cells.loc[:, stages + ["infiltration_factor", "river_bottom"]] = cells.loc[
        :, stages + ["infiltration_factor", "river_bottom"]
    ].mul(cells.loc[:, "conductance_fraction"], axis=0)
    cells = (
        cells.loc[
            :,
            stages
            + ["wet_area", "conductance", "infiltration_factor", "river_bottom"]
            + ["row", "col"],
        ]
        .groupby(["row", "col"])
        .sum()
        .reset_index()
    )

    log.info("select cells where cond > 0")
    cells = cells.loc[cells.loc[:, "conductance"].gt(0.), :]

    log.info("calculate mean stage")
    cells.loc[:, "mean_stage"] = cells.loc[:, stages].mean(axis=1)    

    log.info("exporting to raster")
    for col in cells.columns:
        if col in {"row", "col"}:
            continue
        array = np.empty(domain.shape[-2:])
        array[:] = np.nan
        array[cells.loc[:, "row"], cells.loc[:, "col"]] = cells.loc[:, col]

        # write array to IDF file
        idffile = resultdir / f"river_{col:}.idf"
        log.info(f"writing {idffile:}")

        da = xr.DataArray(array, coords=[domain.coords["y"], domain.coords["x"]])
        imod.idf.save(idffile, da)


def main():
    args = get_parser().parse_args()

    # set logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # arguments from input file
    with open(args.inputfile) as y:
        kwargs = yaml.load(y, yaml.SafeLoader)
        kwargs["inputfile"] = args.inputfile

    # run
    if args.command == "intersect":
        intersect(**kwargs)
    elif args.command == "allocate":
        allocate(**kwargs)
    elif args.command == "export":
        export(**kwargs)
    else:
        raise ValueError(f"unknown command {args.command:}")


if __name__ == "__main__":
    main()
