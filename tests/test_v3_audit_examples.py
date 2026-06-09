import runpy

from pythondea import audit_result, fit, model_catalog
from pythondea.datasets import load_emissions_cross_section


def test_publication_audit_accepts_builtin_sbm_results():
    result = fit("sbm", load_emissions_cross_section())

    audit = audit_result(result)

    assert audit.passed is True
    assert audit.result_hash == result.reproducibility_hash()
    assert {check.name for check in audit.checks} >= {
        "status",
        "tables",
        "primary_table",
        "solver_backend",
        "version",
        "reproducibility_hash",
    }


def test_model_catalog_exposes_citation_metadata():
    catalog = {row["name"]: row for row in model_catalog()}

    assert "sbm" in catalog
    assert catalog["sbm"]["citation_hint"]
    assert catalog["sbm_super_efficiency"]["experimental"] is True


def test_reproduction_example_runs_as_plain_python_script():
    namespace = runpy.run_path("examples/v3_sbm_malmquist_reproduction.py")
    summary = namespace["run"]()

    assert summary["sbm_rows"] == 3
    assert summary["malmquist_rows"] == 2
    assert len(summary["sbm_hash"]) == 64
    assert summary["sbm_audit_passed"] is True
    assert summary["malmquist_audit_passed"] is True


def test_v4_green_productivity_example_runs_as_plain_python_script():
    namespace = runpy.run_path("examples/v4_green_productivity_reproduction.py")
    summary = namespace["run"]()

    assert summary["ddf_rows"] == 2
    assert summary["dirty_beta"] == 0.5
    assert summary["ml_rows"] == 2
    assert summary["ddf_audit_passed"] is True
    assert summary["ml_audit_passed"] is True
