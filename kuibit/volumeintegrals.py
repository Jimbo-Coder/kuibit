#!/usr/bin/env python3

# Copyright (C) 2020-2025 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""Read volume integral output from VolumeIntegrals thorns."""

import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from kuibit import simdir
from kuibit import tensor as kt
from kuibit import timeseries as ts
from kuibit.attr_dict import pythonize_name_dict


_VOLUME_INTEGRALS_FILENAMES = {
    "volume_integrals-GRMHD.asc": "VolumeIntegrals_GRMHD",
    "volume_integrals-vacuum.asc": "VolumeIntegrals_vacuum",
}

_THORN_ORDER = {
    "VolumeIntegrals_GRMHD": 0,
    "VolumeIntegrals_vacuum": 1,
}

# Optional labels for known VolumeIntegrals integrands. Do not use this to
# infer column counts; counts are always parsed from the file header.
_COMPONENT_LABELS = {
    "centerofmass": ("x", "y", "z", "normalization"),
    "centeroflapse": ("x", "y", "z", "normalization"),
    "H_M_CnstraintsL2": ("H", "M0", "M1", "M2"),
    "H_M2_CnstraintsL2": ("H", "M2"),
}

_FLOAT_PATTERN = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"


@dataclass(frozen=True)
class VolumeIntegralHeader:
    """Metadata parsed from one VolumeIntegrals header entry."""

    name: str
    thorn: str
    columns: tuple
    header: str
    region: str
    inside_radius: Optional[float] = None
    outside_radius: Optional[float] = None
    inside_center: Optional[tuple] = None
    outside_center: Optional[tuple] = None
    moves_amr_centre: int = -1
    tracks_amr_centre: int = -1

    @property
    def num_components(self):
        """Return how many output columns belong to this integral."""
        return len(self.columns)

    @property
    def component_labels(self):
        """Return known component labels, if they match the parsed count."""
        labels = _COMPONENT_LABELS.get(self.name)
        if labels is None or len(labels) != self.num_components:
            return None
        return labels

    @property
    def has_amr_centre_interaction(self):
        """Return True if this header has an AMR-centre interaction."""
        return self.moves_amr_centre != -1 or self.tracks_amr_centre != -1

    @property
    def identity_centers(self):
        """Return centers that are stable enough to use for stitching."""
        if self.has_amr_centre_interaction:
            return (None, None)
        return (self.inside_center, self.outside_center)

    @property
    def identity(self):
        """Return the restart-stitching identity for this integral."""
        return (
            self.thorn,
            self.name,
            self.region,
            self.inside_radius,
            self.outside_radius,
            *self.identity_centers,
            self.moves_amr_centre,
            self.tracks_amr_centre,
            self.num_components,
        )

    @property
    def first_column(self):
        """Return the first file column, using zero-based indexing."""
        return self.columns[0]

    def region_description(self):
        """Return a concise description of the integration region."""

        def fmt(value):
            return f"{value:g}"

        if self.region == "full_grid":
            return "full grid"

        parts = []
        if self.inside_radius is not None:
            parts.append(f"inside r={fmt(self.inside_radius)}")
        if self.outside_radius is not None:
            parts.append(f"outside r={fmt(self.outside_radius)}")

        if not self.has_amr_centre_interaction and self.inside_center not in (
            None,
            (0.0, 0.0, 0.0),
        ):
            parts.append(f"inside center={self.inside_center}")
        if (
            not self.has_amr_centre_interaction
            and self.outside_center not in (None, (0.0, 0.0, 0.0))
        ):
            parts.append(f"outside center={self.outside_center}")

        if self.moves_amr_centre != -1:
            parts.append(f"moves AMR {self.moves_amr_centre}")
        if self.tracks_amr_centre != -1:
            parts.append(f"tracks AMR {self.tracks_amr_centre}")

        return " ".join(parts)


def is_volume_integrals(path):
    """Return True if ``path`` looks like a VolumeIntegrals output file."""
    return os.path.basename(str(path)) in _VOLUME_INTEGRALS_FILENAMES


def _parse_center(center):
    """Parse a comma-separated center tuple."""
    return tuple(float(value) for value in center.split(","))


def _parse_header_entry(thorn, column_start, description, column_end):
    """Parse one ``# Col.`` header entry."""
    name, metadata = description.split(".", 1)
    name = name.strip()
    metadata = metadata.strip()

    inside_match = re.search(
        rf"IN sphere @ \(([^)]+)\), r=({_FLOAT_PATTERN})", metadata
    )
    outside_match = re.search(
        rf"OUT sphere @ \(([^)]+)\), r=({_FLOAT_PATTERN})", metadata
    )
    moves_tracks_match = re.search(
        r"Moves/Tracks AMR Centre\s+(-?\d+)/(-?\d+)", metadata
    )

    inside_center = None
    inside_radius = None
    if inside_match is not None:
        inside_center = _parse_center(inside_match.group(1))
        inside_radius = float(inside_match.group(2))

    outside_center = None
    outside_radius = None
    if outside_match is not None:
        outside_center = _parse_center(outside_match.group(1))
        outside_radius = float(outside_match.group(2))

    if inside_match is not None and outside_match is not None:
        region = "shell"
    elif inside_match is not None:
        region = "inside"
    elif outside_match is not None:
        region = "outside"
    elif "(Integral over full grid)" in metadata:
        region = "full_grid"
    else:
        region = "unknown"

    moves_amr_centre = -1
    tracks_amr_centre = -1
    if moves_tracks_match is not None:
        moves_amr_centre = int(moves_tracks_match.group(1))
        tracks_amr_centre = int(moves_tracks_match.group(2))

    columns = tuple(range(column_start - 1, column_end - 1))
    return VolumeIntegralHeader(
        name=name,
        thorn=thorn,
        columns=columns,
        header=description,
        region=region,
        inside_radius=inside_radius,
        outside_radius=outside_radius,
        inside_center=inside_center,
        outside_center=outside_center,
        moves_amr_centre=moves_amr_centre,
        tracks_amr_centre=tracks_amr_centre,
    )


class OneVolumeIntegral:
    """Read one VolumeIntegrals ASCII output file."""

    _rx_column = re.compile(r"^# Col\. (\d+): (.+)$")

    def __init__(self, path):
        """Constructor.

        :param path: Path of the file.
        :type path: str
        """
        self.path = str(path)
        self.folder, filename = os.path.split(self.path)
        if not is_volume_integrals(filename):
            raise RuntimeError(f"Name scheme not recognized for {filename}")

        self.thorn = _VOLUME_INTEGRALS_FILENAMES[filename]
        self.reduction_type = "volume_integrals"
        self._scan_header()

    def _scan_header(self):
        header_entries = []
        first_data = None
        with open(self.path, "r") as file_:
            for line in file_:
                if line.startswith("#"):
                    column_match = self._rx_column.match(line.strip())
                    if column_match is not None:
                        header_entries.append(
                            (
                                int(column_match.group(1)),
                                column_match.group(2),
                            )
                        )
                    continue
                if line.strip():
                    first_data = line
                    break

        if not header_entries:
            raise RuntimeError(f"Unrecognized header in file {self.path}")
        if header_entries[0][0] != 1:
            raise RuntimeError(f"Missing time column in {self.path}")
        if first_data is None:
            raise RuntimeError(f"Missing data in {self.path}")

        num_columns = len(first_data.split())
        data_entries = header_entries[1:]
        self.integrals = []
        for index, (column_start, description) in enumerate(data_entries):
            if index + 1 < len(data_entries):
                column_end = data_entries[index + 1][0]
            else:
                column_end = num_columns + 1
            self.integrals.append(
                _parse_header_entry(
                    self.thorn, column_start, description, column_end
                )
            )

        self._integrals_by_identity = {
            integral.identity: integral for integral in self.integrals
        }
        self._names = {}
        for integral in self.integrals:
            self._names.setdefault(integral.name, []).append(integral)

    @lru_cache(128)
    def load_component(self, identity, component):
        """Read one component as a TimeSeries."""
        integral = self._integrals_by_identity[identity]
        column = integral.columns[component]
        t, y = np.loadtxt(
            self.path,
            unpack=True,
            ndmin=2,
            usecols=(0, column),
        )
        return ts.remove_duplicated_iters(t, y)

    def keys(self):
        """Return the available integral names."""
        return self._names.keys()

    def __contains__(self, key):
        return key in self._names


class VolumeIntegral:
    """One logical volume integral stitched across restarts."""

    def __init__(self, metadata, readers):
        """Constructor.

        :param metadata: Representative metadata.
        :type metadata: :py:class:`~.VolumeIntegralHeader`
        :param readers: Single-file readers for this integral.
        :type readers: list of :py:class:`~.OneVolumeIntegral`
        """
        self.metadata = metadata
        self.readers = readers
        self.name = metadata.name
        self.thorn = metadata.thorn
        self.source = metadata.thorn
        self.region = metadata.region
        self.component_labels = metadata.component_labels
        self._components = {}
        self._vector = None

    def __len__(self):
        return self.metadata.num_components

    def __getitem__(self, component):
        if isinstance(component, str):
            if self.component_labels is None:
                raise KeyError(
                    f"No component labels available for {self.name}"
                )
            try:
                component = self.component_labels.index(component)
            except ValueError as error:
                raise KeyError(f"{component} not available") from error

        if not 0 <= component < len(self):
            raise IndexError(f"Component {component} not available")

        if component not in self._components:
            series = [
                reader.load_component(self.metadata.identity, component)
                for reader in self.readers
            ]
            self._components[component] = ts.combine_ts(series)
        return self._components[component]

    def __getattr__(self, attr):
        if hasattr(kt.Vector, attr) or (
            hasattr(ts.TimeSeries, attr)
            and callable(getattr(ts.TimeSeries, attr))
        ):
            return getattr(self.vector, attr)
        raise AttributeError(f"Object has no attribute {attr}")

    @property
    def t(self):
        """Return the times of the stitched integral."""
        return self[0].t

    @property
    def vector(self):
        """Return components as a :py:class:`~.Vector` of TimeSeries."""
        if self._vector is None:
            self._vector = kt.Vector(
                [self[component] for component in range(len(self))]
            )
        return self._vector

    @property
    def values(self):
        """Return stacked values with shape ``(times, components)``."""
        return np.column_stack(
            [self[component].y for component in range(len(self))]
        )

    def __str__(self):
        component_text = (
            "1 component" if len(self) == 1 else f"{len(self)} components"
        )
        return f"{self.metadata.region_description()}, {component_text}"


class VolumeIntegralOccurrences:
    """List-like collection of occurrences sharing one integral name."""

    def __init__(self, name, occurrences):
        self.name = name
        self._occurrences = list(occurrences)

    def __getitem__(self, index):
        return self._occurrences[index]

    def __iter__(self):
        return iter(self._occurrences)

    def __len__(self):
        return len(self._occurrences)

    def __str__(self):
        return "\n".join(
            f"{self.name}[{index}]: {occurrence}"
            for index, occurrence in enumerate(self._occurrences)
        )


class AllVolumeIntegrals:
    """Read all VolumeIntegrals output in a simulation directory."""

    def __init__(self, allfiles):
        """Constructor.

        :param allfiles: List of all files in the simulation.
        :type allfiles: list of str
        """
        self._integral_readers = {}
        self._representative_metadata = {}

        for file_ in sorted(allfiles):
            if not is_volume_integrals(file_):
                continue
            try:
                volume_integrals_file = OneVolumeIntegral(file_)
            except RuntimeError:
                continue

            for integral in volume_integrals_file.integrals:
                identity = integral.identity
                folder = volume_integrals_file.folder
                self._integral_readers.setdefault(identity, {})
                if folder in self._integral_readers[identity]:
                    warnings.warn(
                        f"Overwriting {integral.name} from "
                        f"{self._integral_readers[identity][folder].path} "
                        f"with {volume_integrals_file.path}",
                        RuntimeWarning,
                    )
                self._integral_readers[identity][
                    folder
                ] = volume_integrals_file
                self._representative_metadata.setdefault(identity, integral)

        self._occurrences = {}
        self._build_occurrences()
        self.fields = pythonize_name_dict(list(self.keys()), self.__getitem__)

    def _build_occurrences(self):
        def sort_key(identity):
            metadata = self._representative_metadata[identity]
            return (
                _THORN_ORDER.get(metadata.thorn, 99),
                metadata.first_column,
                metadata.name,
                metadata.region_description(),
            )

        for identity in sorted(self._integral_readers, key=sort_key):
            metadata = self._representative_metadata[identity]
            readers = [
                self._integral_readers[identity][folder]
                for folder in sorted(self._integral_readers[identity])
            ]
            self._occurrences.setdefault(metadata.name, []).append(
                VolumeIntegral(metadata, readers)
            )

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f"{key} not available")
        return VolumeIntegralOccurrences(key, self._occurrences[key])

    def __contains__(self, key):
        return key in self._occurrences

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"Object has no attribute {key}")

    def __len__(self):
        return len(self._occurrences)

    def keys(self):
        """Return the available volume integral names."""
        return self._occurrences.keys()

    def get(self, key, default=None):
        """Return integral occurrences if available, else return default."""
        if key in self:
            return self[key]
        return default

    def __str__(self):
        if len(self) == 0:
            return "No volume integrals found"

        lines = ["Available volume integrals:"]
        for thorn in sorted(
            {occ.thorn for occs in self._occurrences.values() for occ in occs},
            key=lambda thorn: _THORN_ORDER.get(thorn, 99),
        ):
            lines.append(f"{thorn}:")
            for name in self.keys():
                for index, occurrence in enumerate(self._occurrences[name]):
                    if occurrence.thorn != thorn:
                        continue
                    lines.append(f"  {name}[{index}]: {occurrence}")
        return "\n".join(lines)


class VolumeIntegralsDir(AllVolumeIntegrals):
    """Interface to VolumeIntegrals output in a simulation directory."""

    def __init__(self, sd):
        """Constructor.

        :param sd: Simulation directory.
        :type sd: :py:class:`~.SimDir`
        """
        if not isinstance(sd, simdir.SimDir):
            raise TypeError("Input is not SimDir")

        self.path = sd.path
        super().__init__(sd.allfiles)
