from pathlib import Path
from typing import List

import plotly.graph_objects as go


def write_figs(figs: List[go.Figure], out_path: Path) -> None:
    include_js = True
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path.expanduser(), "w") as f:
        f.write("<html><head></head><body>\n")
        for fig in figs:
            fig.write_html(
                f,
                include_plotlyjs=include_js,
                full_html=False,
                auto_open=False,
                default_height="60%",
            )
            include_js = False
        f.write("</body></html>\n")
