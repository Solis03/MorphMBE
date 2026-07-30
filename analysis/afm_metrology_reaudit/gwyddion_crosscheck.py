from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


GWY_RUN_NONINTERACTIVE = 1
GWY_MASK_IGNORE = 2
GWY_SI_UNIT_FORMAT_PLAIN = 1
NM_PER_M = 1.0e9


@dataclass(frozen=True)
class GwyddionChannelResult:
    raw_path: str
    channel_id: int
    channel_title: str
    xres: int
    yres: int
    z_unit: str
    raw_rms_nm: float
    line0_sq_nm: float
    line1_sq_nm: float
    line2_sq_nm: float
    line3_sq_nm: float


class GwyddionAPI:
    """Minimal ctypes bridge to the locally installed Gwyddion libraries.

    This invokes Gwyddion's own NanoScope importer, row polynomial levelling,
    and RMS implementation.  It is intentionally independent of NumPy's AFM
    decoder and line-flattening implementation used by the model pipeline.
    """

    def __init__(self, prefix: str | Path = "/opt/homebrew") -> None:
        prefix = Path(prefix)
        self.prefix = prefix
        mode = ctypes.RTLD_GLOBAL
        self.glib = ctypes.CDLL(
            str(prefix / "opt/glib/lib/libglib-2.0.0.dylib"),
            mode=mode,
        )
        self.gobject = ctypes.CDLL(
            str(prefix / "opt/glib/lib/libgobject-2.0.0.dylib"),
            mode=mode,
        )
        self.gwyddion = ctypes.CDLL(
            str(prefix / "lib/libgwyddion2.dylib"),
            mode=mode,
        )
        self.process = ctypes.CDLL(
            str(prefix / "lib/libgwyprocess2.dylib"),
            mode=mode,
        )
        # The module library exposes graph-aware file containers and the
        # Homebrew build resolves these types through the draw/widget
        # libraries at runtime.
        self.draw = ctypes.CDLL(
            str(prefix / "lib/libgwydraw2.dylib"),
            mode=mode,
        )
        self.dgets = ctypes.CDLL(
            str(prefix / "lib/libgwydgets2.dylib"),
            mode=mode,
        )
        self.module = ctypes.CDLL(
            str(prefix / "lib/libgwymodule2.dylib"),
            mode=mode,
        )
        self.app = ctypes.CDLL(
            str(prefix / "lib/libgwyapp2.dylib"),
            mode=mode,
        )
        self._declare_signatures()
        # This is Gwyddion's documented batch-program initialiser.  It sets
        # up types, resources, settings and file modules without opening GTK
        # or requiring a display.
        self.app.gwy_app_init_nongui(None)

    def _declare_signatures(self) -> None:
        self.glib.g_quark_try_string.argtypes = [ctypes.c_char_p]
        self.glib.g_quark_try_string.restype = ctypes.c_uint32
        self.glib.g_quark_from_string.argtypes = [ctypes.c_char_p]
        self.glib.g_quark_from_string.restype = ctypes.c_uint32
        self.glib.g_free.argtypes = [ctypes.c_void_p]
        self.glib.g_free.restype = None
        self.gobject.g_object_unref.argtypes = [ctypes.c_void_p]
        self.gobject.g_object_unref.restype = None

        self.app.gwy_app_init_nongui.restype = None
        self.module.gwy_file_load.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.module.gwy_file_load.restype = ctypes.c_void_p

        self.gwyddion.gwy_container_get_object.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.gwyddion.gwy_container_get_object.restype = ctypes.c_void_p
        self.gwyddion.gwy_container_contains.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.gwyddion.gwy_container_contains.restype = ctypes.c_int
        self.gwyddion.gwy_container_gis_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.gwyddion.gwy_container_gis_string.restype = ctypes.c_int
        self.gwyddion.gwy_serializable_duplicate.argtypes = [ctypes.c_void_p]
        self.gwyddion.gwy_serializable_duplicate.restype = ctypes.c_void_p
        self.gwyddion.gwy_si_unit_get_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.gwyddion.gwy_si_unit_get_string.restype = ctypes.c_void_p

        self.process.gwy_data_field_get_xres.argtypes = [ctypes.c_void_p]
        self.process.gwy_data_field_get_xres.restype = ctypes.c_int
        self.process.gwy_data_field_get_yres.argtypes = [ctypes.c_void_p]
        self.process.gwy_data_field_get_yres.restype = ctypes.c_int
        self.process.gwy_data_field_get_si_unit_z.argtypes = [ctypes.c_void_p]
        self.process.gwy_data_field_get_si_unit_z.restype = ctypes.c_void_p
        self.process.gwy_data_field_get_rms.argtypes = [ctypes.c_void_p]
        self.process.gwy_data_field_get_rms.restype = ctypes.c_double
        self.process.gwy_data_field_get_data_const.argtypes = [ctypes.c_void_p]
        self.process.gwy_data_field_get_data_const.restype = ctypes.POINTER(
            ctypes.c_double
        )
        self.process.gwy_data_field_area_extract.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.process.gwy_data_field_area_extract.restype = ctypes.c_void_p
        self.process.gwy_data_field_row_level_poly.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.process.gwy_data_field_row_level_poly.restype = None

    def _quark(self, key: str, *, create: bool = False) -> int:
        function = (
            self.glib.g_quark_from_string
            if create
            else self.glib.g_quark_try_string
        )
        return int(function(key.encode("utf-8")))

    def _object(self, container: int, key: str) -> int | None:
        quark = self._quark(key)
        if not quark:
            return None
        if not self.gwyddion.gwy_container_contains(container, quark):
            return None
        value = self.gwyddion.gwy_container_get_object(container, quark)
        return int(value) if value else None

    def _string(self, container: int, key: str) -> str:
        quark = self._quark(key)
        if not quark:
            return ""
        value = ctypes.c_char_p()
        found = self.gwyddion.gwy_container_gis_string(
            container,
            quark,
            ctypes.byref(value),
        )
        if not found or not value.value:
            return ""
        return value.value.decode("utf-8", errors="replace")

    def _unit_string(self, field: int) -> str:
        unit = self.process.gwy_data_field_get_si_unit_z(field)
        allocated = self.gwyddion.gwy_si_unit_get_string(
            unit,
            GWY_SI_UNIT_FORMAT_PLAIN,
        )
        if not allocated:
            return ""
        try:
            return ctypes.string_at(allocated).decode(
                "utf-8",
                errors="replace",
            )
        finally:
            self.glib.g_free(allocated)

    def channel_array_nm(self, field: int) -> np.ndarray:
        xres = int(self.process.gwy_data_field_get_xres(field))
        yres = int(self.process.gwy_data_field_get_yres(field))
        pointer = self.process.gwy_data_field_get_data_const(field)
        return (
            np.ctypeslib.as_array(pointer, shape=(xres * yres,))
            .reshape(yres, xres)
            .copy()
            * NM_PER_M
        )

    def inspect(self, raw_path: str | Path) -> list[GwyddionChannelResult]:
        raw_path = Path(raw_path).resolve()
        error = ctypes.c_void_p()
        container = self.module.gwy_file_load(
            str(raw_path).encode("utf-8"),
            GWY_RUN_NONINTERACTIVE,
            ctypes.byref(error),
        )
        if not container:
            raise RuntimeError(
                f"Gwyddion could not load {raw_path}; GError={error.value}"
            )
        records: list[GwyddionChannelResult] = []
        try:
            for channel_id in range(64):
                field = self._object(container, f"/{channel_id}/data")
                if field is None:
                    continue
                title = self._string(container, f"/{channel_id}/data/title")
                raw_rms = (
                    float(self.process.gwy_data_field_get_rms(field))
                    * NM_PER_M
                )
                levelled: dict[int, float] = {}
                for order in range(4):
                    duplicate = self.gwyddion.gwy_serializable_duplicate(field)
                    if not duplicate:
                        raise RuntimeError("Gwyddion field duplication failed")
                    try:
                        self.process.gwy_data_field_row_level_poly(
                            duplicate,
                            None,
                            GWY_MASK_IGNORE,
                            order,
                            None,
                        )
                        levelled[order] = (
                            float(
                                self.process.gwy_data_field_get_rms(duplicate)
                            )
                            * NM_PER_M
                        )
                    finally:
                        self.gobject.g_object_unref(duplicate)
                records.append(
                    GwyddionChannelResult(
                        raw_path=str(raw_path),
                        channel_id=channel_id,
                        channel_title=title,
                        xres=int(
                            self.process.gwy_data_field_get_xres(field)
                        ),
                        yres=int(
                            self.process.gwy_data_field_get_yres(field)
                        ),
                        z_unit=self._unit_string(field),
                        raw_rms_nm=raw_rms,
                        line0_sq_nm=levelled[0],
                        line1_sq_nm=levelled[1],
                        line2_sq_nm=levelled[2],
                        line3_sq_nm=levelled[3],
                    )
                )
        finally:
            self.gobject.g_object_unref(container)
        return records

    def subfield_line_sq_nm(
        self,
        raw_path: str | Path,
        *,
        col: int,
        row: int,
        width: int,
        height: int,
        channel_title: str = "ZSensor",
    ) -> dict[int, float]:
        """Return Gwyddion line-levelled Sq for one imported AFM subfield."""

        raw_path = Path(raw_path).resolve()
        error = ctypes.c_void_p()
        container = self.module.gwy_file_load(
            str(raw_path).encode("utf-8"),
            GWY_RUN_NONINTERACTIVE,
            ctypes.byref(error),
        )
        if not container:
            raise RuntimeError(
                f"Gwyddion could not load {raw_path}; GError={error.value}"
            )
        try:
            field = None
            available: list[str] = []
            for channel_id in range(64):
                candidate = self._object(container, f"/{channel_id}/data")
                if candidate is None:
                    continue
                title = self._string(
                    container,
                    f"/{channel_id}/data/title",
                )
                available.append(title)
                if title == channel_title:
                    field = candidate
                    break
            if field is None:
                raise RuntimeError(
                    f"{channel_title!r} is absent from {raw_path}; "
                    f"available={available}"
                )
            extracted = self.process.gwy_data_field_area_extract(
                field,
                int(col),
                int(row),
                int(width),
                int(height),
            )
            if not extracted:
                raise RuntimeError(
                    f"Gwyddion could not extract ({col}, {row}, "
                    f"{width}, {height}) from {raw_path}"
                )
            try:
                results: dict[int, float] = {}
                for order in range(4):
                    duplicate = self.gwyddion.gwy_serializable_duplicate(
                        extracted
                    )
                    if not duplicate:
                        raise RuntimeError(
                            "Gwyddion field duplication failed"
                        )
                    try:
                        self.process.gwy_data_field_row_level_poly(
                            duplicate,
                            None,
                            GWY_MASK_IGNORE,
                            order,
                            None,
                        )
                        results[order] = (
                            float(
                                self.process.gwy_data_field_get_rms(duplicate)
                            )
                            * NM_PER_M
                        )
                    finally:
                        self.gobject.g_object_unref(duplicate)
                return results
            finally:
                self.gobject.g_object_unref(extracted)
        finally:
            self.gobject.g_object_unref(container)


def _write_csv(records: Iterable[GwyddionChannelResult], path: Path) -> None:
    import csv

    rows = [asdict(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Gwyddion audit produced no channels")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check NanoScope AFM with Gwyddion 2.71",
    )
    parser.add_argument("raw_paths", nargs="*")
    parser.add_argument(
        "--path-table",
        action="append",
        type=Path,
        default=[],
        help="CSV table containing raw AFM paths (repeatable)",
    )
    parser.add_argument("--path-column", default="raw_afm_file")
    parser.add_argument(
        "--path-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve relative paths read from CSV tables",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--gwyddion-prefix", default="/opt/homebrew")
    args = parser.parse_args()
    raw_paths = list(args.raw_paths)
    for table_path in args.path_table:
        with table_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = str(row.get(args.path_column, "")).strip()
                if value:
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = args.path_root / candidate
                    raw_paths.append(str(candidate))
    raw_paths = list(dict.fromkeys(raw_paths))
    if not raw_paths:
        parser.error("provide raw_paths or at least one --path-table")
    api = GwyddionAPI(args.gwyddion_prefix)
    records = [
        result
        for raw_path in raw_paths
        for result in api.inspect(raw_path)
    ]
    if args.output_csv:
        _write_csv(records, args.output_csv)
    payload = [
        {
            key: _finite(value) if isinstance(value, float) else value
            for key, value in asdict(record).items()
        }
        for record in records
    ]
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
