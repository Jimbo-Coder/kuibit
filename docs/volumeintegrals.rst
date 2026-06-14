VolumeIntegrals data
====================

The module :py:mod:`~.volumeintegrals`
(:ref:`volumeintegrals_ref:Reference on kuibit.volumeintegrals`) reads output
from the ``VolumeIntegrals_GRMHD`` and ``VolumeIntegrals_vacuum`` thorns. Data
is loaded lazily and restarts are stitched with the same time-series machinery
used elsewhere in ``kuibit``.

Accessing data
--------------

Volume integrals are available from :py:class:`~kuibit.simdir.SimDir`:

.. code-block:: python

    import kuibit.simdir as sd

    sim = sd.SimDir("simulation")

    volints = sim.volumeintegrals
    # or
    volints = sim.volints

Integrals are organized by their header name. Repeated integrals are stored as
occurrences:

.. code-block:: python

    restmass = sim.volints["restmass"][0]
    same = sim.volints.restmass[0]
    same_again = sim.volints.fields.restmass[0]

    time = restmass.t
    values = restmass.values

``values`` has shape ``(times, components)``. For one-column integrals this is a
single column. Components can also be accessed as
:py:class:`~kuibit.timeseries.TimeSeries` objects:

.. code-block:: python

    restmass_ts = restmass[0]

Known multi-column integrals expose component labels when available:

.. code-block:: python

    center = sim.volints.centerofmass[0]
    x = center["x"]
    normalization = center["normalization"]

Metadata
--------

Each occurrence stores the parsed header metadata in ``metadata``. This includes
the thorn, integration region, inner and outer radii, centers, number of
components, and AMR-center tracking fields.

``moves_amr_centre`` and ``tracks_amr_centre`` use the values printed by the
VolumeIntegrals header: ``-1`` means disabled, otherwise the value is the AMR
center index.

Restart handling
----------------

Individual files are cleaned with
:py:func:`~kuibit.timeseries.remove_duplicated_iters`. Multiple outputs are
combined with :py:func:`~kuibit.timeseries.combine_ts`, preferring later restart
segments in overlapping intervals.
