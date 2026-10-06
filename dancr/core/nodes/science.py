"""Read scientific array formats: NetCDF (``load_netcdf``) and HDF5 (``load_hdf5``).

Both use xarray (NetCDF) and h5py (HDF5), which ship with DANCR, and turn one variable or
dataset into an ordinary table, so everything downstream (time steps, charts, reports) works as usual. A plain
message asks for the extra when it is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry


def _need(package: str, extra: str | None = None) -> None:
    raise ValueError(f"This build is missing the '{package}' package needed to read this format. "
                     "Reinstall DANCR (or, in a source checkout, run 'uv sync').")


def _load_netcdf(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    raw = str(params.get("path") or "").strip()
    if not raw:
        raise ValueError("Choose a NetCDF file")
    path = ctx.resolve(raw)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import xarray as xr
    except ImportError:
        _need("xarray", "science")
    try:
        ds = xr.open_dataset(path)
    except Exception as e:  # noqa: BLE001 - report the file, not a driver trace
        raise ValueError(f"Cannot read {path.name}: {e}") from e
    name = str(params.get("variable") or "").strip()
    if not name:
        variables = list(ds.data_vars)
        if not variables:
            raise ValueError(f"{path.name} has no data variables")
        name = variables[0]
    if name not in ds:
        raise ValueError(f"{path.name} has no variable {name!r}. Variables: {', '.join(ds.data_vars) or 'none'}")
    da = ds[name]
    try:
        df = da.to_dataframe(name=name).reset_index()
    finally:
        ds.close()
    out = pl.from_pandas(df)
    return NodeResult(out.lazy(), messages=[f"Read '{name}' from {path.name} ({out.height:,} rows × {out.width} columns)"],
                      report={"variable": name, "rows": out.height})


registry.register(NodeType(
    key="load_netcdf", label="Load NetCDF", category="Get data", icon="wave-sine", kind="source", inputs=[],
    description="Read one variable of a NetCDF file as a table (uses xarray, included).",
    apply=_load_netcdf, summary=lambda p: Path(str(p.get("path") or "")).name or "no file",
    params=[
        Param("path", "File", "path", required=True, help="A .nc / .nc4 NetCDF file"),
        Param("variable", "Variable", "text", default="", help="Leave blank for the first one"),
    ],
))


def _load_hdf5(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    raw = str(params.get("path") or "").strip()
    dataset = str(params.get("dataset") or "").strip()
    if not raw:
        raise ValueError("Choose an HDF5 file")
    if not dataset:
        raise ValueError("Name the dataset inside the file")
    path = ctx.resolve(raw)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import h5py
        import numpy as np
    except ImportError:
        _need("h5py", "science")
    try:
        with h5py.File(path, "r") as f:
            if dataset not in f:
                keys = list(f.keys())
                raise ValueError(f"{path.name} has no dataset {dataset!r}. Top-level keys: {', '.join(keys) or 'none'}")
            arr = np.asarray(f[dataset][...])
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Cannot read {path.name}: {e}") from e
    if arr.ndim == 1:
        out = pl.DataFrame({"value": arr})
    elif arr.ndim == 2:
        out = pl.DataFrame({f"col_{i}": arr[:, i] for i in range(arr.shape[1])})
    else:
        raise ValueError(f"{dataset} is {arr.ndim}-dimensional; only one- and two-dimensional datasets become a table")
    return NodeResult(out.lazy(), messages=[f"Read '{dataset}' from {path.name} ({out.height:,} rows × {out.width} columns)"],
                      report={"dataset": dataset, "rows": out.height})


registry.register(NodeType(
    key="load_hdf5", label="Load HDF5", category="Get data", icon="columns", kind="source", inputs=[],
    description="Read one dataset of an HDF5 file as a table (uses h5py, included).",
    apply=_load_hdf5, summary=lambda p: Path(str(p.get("path") or "")).name or "no file",
    params=[
        Param("path", "File", "path", required=True, help="An .h5 / .hdf5 file"),
        Param("dataset", "Dataset path", "text", required=True, placeholder="/group/dataset"),
    ],
))
