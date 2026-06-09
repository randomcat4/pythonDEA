"""Small deterministic datasets used by docs and smoke tests."""

from __future__ import annotations

from .core.data import DEAData, PanelDEAData


def load_emissions_cross_section() -> DEAData:
    """Return a tiny bad-output DEA dataset."""

    return DEAData(
        inputs=[[1.0], [1.0], [1.0]],
        good_outputs=[[1.0], [1.0], [1.0]],
        bad_outputs=[[1.0], [1.2], [2.2]],
        dmu_names=["clean", "balanced", "dirty"],
        input_names=["capital"],
        good_output_names=["output"],
        bad_output_names=["emissions"],
    )


def load_productivity_panel() -> PanelDEAData:
    """Return a two-period panel for SBM-Malmquist examples."""

    return PanelDEAData.from_3d(
        periods=["2020", "2021"],
        entities=["A", "B"],
        inputs=[
            [[1.0], [2.0]],
            [[0.9], [2.1]],
        ],
        good_outputs=[
            [[1.0], [1.0]],
            [[1.1], [1.1]],
        ],
        input_names=["labor"],
        good_output_names=["service"],
    )
