"""Minimal v4 DDF and Malmquist-Luenberger reproduction script."""

from pythondea import DEAData, PanelDEAData, audit_result, fit


def run():
    cross_section = DEAData(
        inputs=[[1.0], [1.0]],
        good_outputs=[[1.0], [1.0]],
        bad_outputs=[[1.0], [2.0]],
        dmu_names=["clean", "dirty"],
        bad_output_names=["co2"],
    )
    panel = PanelDEAData.from_3d(
        periods=["2020", "2021"],
        entities=["A", "B"],
        inputs=[[[1.0], [1.0]], [[1.0], [1.0]]],
        good_outputs=[[[1.0], [1.0]], [[1.0], [1.0]]],
        bad_outputs=[[[1.0], [2.0]], [[1.0], [1.5]]],
        bad_output_names=["co2"],
    )
    ddf = fit("directional_distance", cross_section, direction="bad_output")
    ml = fit("malmquist_luenberger", panel, direction="bad_output")
    return {
        "ddf_rows": len(ddf.rows),
        "dirty_beta": [row for row in ddf.rows if row["dmu"] == "dirty"][0]["beta"],
        "ml_rows": len(ml.rows),
        "ddf_audit_passed": audit_result(ddf).passed,
        "ml_audit_passed": audit_result(ml).passed,
    }


if __name__ == "__main__":
    summary = run()
    for key, value in summary.items():
        print(f"{key}: {value}")
