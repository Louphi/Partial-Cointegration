"""Fresh-kernel notebook verification and a real Streamlit app flow."""

import json
from pathlib import Path
import sys
import os
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
tmp = root / ".test-deps/jupyter"
kernels = tmp / "kernels/pci"
kernels.mkdir(parents=True, exist_ok=True)
(kernels / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "PCI audit Python",
            "language": "python",
        }
    )
)
os.environ["JUPYTER_PATH"] = str(tmp)
os.environ["JUPYTER_RUNTIME_DIR"] = str(tmp / "runtime")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
report = []
for path in ["Simulation/Functions/HowToUse.ipynb", "PCI Framework/VBT_TMP.ipynb"]:
    notebook = nbformat.read(root / path, as_version=4)
    for i, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{i}"
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="pci",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    dest = root / ".test-deps/executed" / Path(path).name
    dest.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, dest)
    report.append(
        {
            "notebook": path,
            "executed_cells": sum(c.cell_type == "code" for c in notebook.cells),
            "status": "passed",
        }
    )
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(root / "PCI Framework/app.py"), default_timeout=60)
app.run()
assert not app.exception, app.exception
app.button[0].click().run()
assert not app.exception, app.exception
assert not app.error, app.error
report.append(
    {
        "application": "Streamlit synthetic demonstration",
        "action": "click Run research example",
        "status": "passed",
    }
)
import pandas as pd
from partial_cointegration import Parameters
from partial_cointegration.simulation import simulate_pair

pair = simulate_pair(100, Parameters(1, 0.6, 1, 1), seed=40)
prices = pd.DataFrame(
    {"X": pair.x, "Y": pair.y}, index=pd.bdate_range("2020-01-01", periods=100)
)
fixture = root / ".test-deps/app-price-fixture.csv"
prices.to_csv(fixture, index_label="date")
configured = AppTest.from_file(
    str(root / "PCI Framework/app.py"), default_timeout=60
).run()
configured.selectbox[0].set_value("Historical CSV/parquet").run()
configured.text_input[0].input(str(fixture)).run()
assert not configured.exception, configured.exception
assert not configured.error, configured.error
configured.button[0].click().run()
assert not configured.exception, configured.exception
assert not configured.error, configured.error
prices.iloc[:20].to_csv(fixture, index_label="date")
configured.run()
assert not configured.exception, configured.exception
assert configured.error and "30 observations" in configured.error[0].value
report.append(
    {
        "application": "Streamlit configured CSV flow using synthetic fixture",
        "action": "load CSV, run analysis, reject too-short data cleanly",
        "status": "passed",
    }
)
(root / "docs/results/notebook-app-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
