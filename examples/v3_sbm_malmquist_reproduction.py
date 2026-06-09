"""Minimal v3 SBM and SBM-Malmquist reproduction script."""

from pythondea import audit_result, fit
from pythondea.datasets import load_emissions_cross_section, load_productivity_panel


def run():
    sbm = fit("sbm", load_emissions_cross_section(), orientation="bad_output_adjusted")
    malmquist = fit("sbm_malmquist", load_productivity_panel())
    return {
        "sbm_rows": len(sbm.rows),
        "malmquist_rows": len(malmquist.rows),
        "sbm_hash": sbm.reproducibility_hash(),
        "malmquist_hash": malmquist.reproducibility_hash(),
        "sbm_audit_passed": audit_result(sbm).passed,
        "malmquist_audit_passed": audit_result(malmquist).passed,
    }


if __name__ == "__main__":
    summary = run()
    for key, value in summary.items():
        print(f"{key}: {value}")
