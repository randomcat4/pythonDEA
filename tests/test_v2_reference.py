from v2 import (
    PanelDEAData,
    contemporaneous_reference,
    cross_period_reference,
    global_reference,
    window_reference,
)


def _panel():
    return PanelDEAData.from_3d(
        periods=["2020", "2021", "2022"],
        entities=["A", "B"],
        inputs=[
            [[1.0], [2.0]],
            [[1.1], [2.1]],
            [[1.2], [2.2]],
        ],
        good_outputs=[
            [[1.0], [1.0]],
            [[1.2], [1.1]],
            [[1.3], [1.2]],
        ],
    )


def test_panel_flattens_balanced_rows_with_stable_names():
    panel = _panel()

    assert panel.data.dmu_names == [
        "A@2020",
        "B@2020",
        "A@2021",
        "B@2021",
        "A@2022",
        "B@2022",
    ]
    assert panel.row_index("2021", "B") == 3


def test_contemporaneous_and_global_reference_indices():
    panel = _panel()

    same_period = contemporaneous_reference(panel, "2021")
    global_frontier = global_reference(panel)

    assert same_period.indices == (2, 3)
    assert same_period.label == "period:2021"
    assert global_frontier.indices == (0, 1, 2, 3, 4, 5)
    assert global_frontier.label == "global"


def test_cross_period_reference_indices_for_previous_and_next_frontiers():
    panel = _panel()

    previous_frontier = cross_period_reference(panel, "2020")
    next_frontier = cross_period_reference(panel, "2022")

    assert previous_frontier.indices == (0, 1)
    assert previous_frontier.label == "cross-period:2020"
    assert next_frontier.indices == (4, 5)
    assert next_frontier.label == "cross-period:2022"


def test_window_reference_indices_are_closed_range():
    panel = _panel()

    reference = window_reference(panel, "2020", "2021")

    assert reference.indices == (0, 1, 2, 3)
    assert reference.label == "window:2020:2021"
