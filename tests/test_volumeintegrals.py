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

import unittest

import numpy as np

from kuibit import simdir as sd
from kuibit import tensor as kt
from kuibit import timeseries as ts
from kuibit import volumeintegrals as vi


class TestVolumeIntegrals(unittest.TestCase):
    def test_OneVolumeIntegral(self):
        with self.assertRaises(RuntimeError):
            vi.OneVolumeIntegral("not-volume-integrals.asc")

        reader = vi.OneVolumeIntegral(
            "tests/volume_integrals/output-0000/volume_integration/"
            "volume_integrals-GRMHD.asc"
        )

        self.assertEqual(reader.thorn, "VolumeIntegrals_GRMHD")
        self.assertCountEqual(
            reader.keys(),
            ["one", "centerofmass", "usepreviousintegrands", "restmass"],
        )
        self.assertEqual(len(reader.integrals), 12)

        centerofmass = reader.integrals[1]
        self.assertEqual(centerofmass.name, "centerofmass")
        self.assertEqual(centerofmass.num_components, 4)
        self.assertTupleEqual(
            centerofmass.component_labels,
            ("x", "y", "z", "normalization"),
        )

        shell = reader.integrals[7]
        self.assertEqual(shell.name, "one")
        self.assertEqual(shell.region, "shell")
        self.assertEqual(shell.inside_radius, 16.0)
        self.assertEqual(shell.outside_radius, 4.0)
        self.assertEqual(shell.moves_amr_centre, -1)
        self.assertEqual(shell.tracks_amr_centre, -1)

        moves_amr = reader.integrals[8]
        self.assertEqual(moves_amr.name, "one")
        self.assertEqual(moves_amr.region, "inside")
        self.assertEqual(moves_amr.inside_radius, 10.0)
        self.assertEqual(moves_amr.moves_amr_centre, 0)
        self.assertEqual(moves_amr.tracks_amr_centre, -1)

        tracks_amr = reader.integrals[9]
        self.assertEqual(tracks_amr.name, "one")
        self.assertEqual(tracks_amr.region, "inside")
        self.assertEqual(tracks_amr.inside_radius, 14.0)
        self.assertEqual(tracks_amr.moves_amr_centre, -1)
        self.assertEqual(tracks_amr.tracks_amr_centre, 0)

        centerofmass_moves_amr = reader.integrals[10]
        self.assertEqual(centerofmass_moves_amr.name, "centerofmass")
        self.assertEqual(centerofmass_moves_amr.num_components, 4)
        self.assertEqual(centerofmass_moves_amr.inside_radius, 18.0)
        self.assertEqual(centerofmass_moves_amr.moves_amr_centre, 0)
        self.assertEqual(centerofmass_moves_amr.tracks_amr_centre, -1)

        centerofmass_moves_and_tracks_amr = reader.integrals[11]
        self.assertEqual(
            centerofmass_moves_and_tracks_amr.name, "centerofmass"
        )
        self.assertEqual(centerofmass_moves_and_tracks_amr.region, "shell")
        self.assertEqual(centerofmass_moves_and_tracks_amr.inside_radius, 20.0)
        self.assertEqual(centerofmass_moves_and_tracks_amr.outside_radius, 2.0)
        self.assertEqual(centerofmass_moves_and_tracks_amr.moves_amr_centre, 0)
        self.assertEqual(
            centerofmass_moves_and_tracks_amr.tracks_amr_centre, 0
        )
        self.assertEqual(centerofmass_moves_and_tracks_amr.num_components, 4)

    def test_AllVolumeIntegrals(self):
        sim = sd.SimDir("tests/volume_integrals")

        self.assertIs(sim.volumeintegrals, sim.volints)
        volints = sim.volints

        self.assertIn("one", volints)
        self.assertIn("restmass", volints)
        self.assertIn("ADM_Mass", volints)
        self.assertCountEqual(
            volints.keys(),
            [
                "one",
                "centerofmass",
                "usepreviousintegrands",
                "restmass",
                "ADM_Mass",
                "ADM_Momentum",
                "ADM_Angular_Momentum",
                "H_M_CnstraintsL2",
                "H_M2_CnstraintsL2",
                "centeroflapse",
            ],
        )

        ones = volints["one"]
        self.assertEqual(len(ones), 12)
        self.assertEqual(ones[0].thorn, "VolumeIntegrals_GRMHD")
        self.assertEqual(ones[0].region, "full_grid")
        self.assertEqual(ones[4].region, "shell")
        self.assertEqual(ones[5].metadata.moves_amr_centre, 0)
        self.assertEqual(ones[5].metadata.tracks_amr_centre, -1)
        self.assertEqual(ones[6].metadata.moves_amr_centre, -1)
        self.assertEqual(ones[6].metadata.tracks_amr_centre, 0)
        self.assertEqual(ones[7].thorn, "VolumeIntegrals_vacuum")
        self.assertEqual(ones[7].region, "full_grid")

        restmass = volints["restmass"][0]
        self.assertEqual(len(restmass), 1)
        self.assertTrue(np.allclose(restmass.t, [0.0, 0.2, 0.4, 0.6, 0.8]))
        self.assertEqual(restmass.values.shape, (5, 1))
        self.assertIsInstance(restmass[0], ts.TimeSeries)
        self.assertIs(volints.restmass[0], restmass)

        centerofmass = volints["centerofmass"][0]
        self.assertEqual(len(centerofmass), 4)
        self.assertEqual(centerofmass.values.shape, (5, 4))
        self.assertEqual(centerofmass["x"], centerofmass[0])
        self.assertIsInstance(centerofmass.vector, kt.Vector)
        self.assertEqual(centerofmass.vector[0], centerofmass[0])
        self.assertTrue(
            np.allclose(
                centerofmass.norm().y,
                np.sqrt(np.sum(centerofmass.values**2, axis=1)),
            )
        )
        self.assertIsInstance(centerofmass.differentiated(), kt.Vector)
        self.assertTrue(
            np.allclose(centerofmass.values[:, 0], centerofmass[0].y)
        )

        centerofmass_moves_amr = volints["centerofmass"][1]
        self.assertEqual(centerofmass_moves_amr.metadata.moves_amr_centre, 0)
        self.assertEqual(centerofmass_moves_amr.metadata.tracks_amr_centre, -1)
        self.assertEqual(centerofmass_moves_amr.metadata.inside_radius, 18.0)
        self.assertEqual(centerofmass_moves_amr.values.shape, (5, 4))

        centerofmass_moves_and_tracks_amr = volints["centerofmass"][2]
        self.assertEqual(
            centerofmass_moves_and_tracks_amr.metadata.moves_amr_centre, 0
        )
        self.assertEqual(
            centerofmass_moves_and_tracks_amr.metadata.tracks_amr_centre, 0
        )
        self.assertEqual(
            centerofmass_moves_and_tracks_amr.metadata.inside_radius, 20.0
        )
        self.assertEqual(
            centerofmass_moves_and_tracks_amr.metadata.outside_radius, 2.0
        )
        self.assertTrue(
            np.allclose(
                centerofmass_moves_and_tracks_amr.t,
                [0.0, 0.2, 0.4, 0.6, 0.8],
            )
        )
        self.assertEqual(
            centerofmass_moves_and_tracks_amr.values.shape, (5, 4)
        )

        h_m2 = volints["H_M2_CnstraintsL2"][0]
        self.assertEqual(len(h_m2), 2)
        self.assertEqual(h_m2.component_labels, ("H", "M2"))
        self.assertEqual(h_m2["M2"], h_m2[1])

        self.assertIs(volints.fields.restmass[0], restmass)
        with self.assertRaises(AttributeError):
            volints.not_an_integral

    def test_str(self):
        volints = sd.SimDir("tests/volume_integrals").volints
        string = str(volints)

        self.assertIn("VolumeIntegrals_GRMHD:", string)
        self.assertIn("VolumeIntegrals_vacuum:", string)
        self.assertIn("one[0]: full grid, 1 component", string)
        self.assertIn("one[5]: inside r=10 moves AMR 0, 1 component", string)
        self.assertIn("one[6]: inside r=14 tracks AMR 0, 1 component", string)
        self.assertIn(
            "centerofmass[1]: inside r=18 moves AMR 0, 4 components",
            string,
        )
        self.assertIn(
            "centerofmass[2]: inside r=20 outside r=2",
            string,
        )
        self.assertIn("moves AMR 0 tracks AMR 0, 4 components", string)
        self.assertIn("one[7]: full grid, 1 component", string)
        self.assertIn("H_M2_CnstraintsL2[0]: full grid, 2 components", string)


if __name__ == "__main__":
    unittest.main()
