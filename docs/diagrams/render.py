"""Render the Mermaid diagrams in this folder to PNG, one light and one dark version each.

    uv run python docs/diagrams/render.py

GitHub's live Mermaid renderer draws labels as HTML, which some browsers clip. Static images look
the same everywhere; the README picks the light or dark one with a <picture> tag.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"  # single-file build, fetched once
THEMES = {"light": "default", "dark": "dark"}

RUN = """async ([theme, config]) => {
  mermaid.initialize({startOnLoad: false, theme, securityLevel: "strict", ...config});
  try { await mermaid.run(); return "OK"; } catch (e) { return "ERR " + e.message; }
}"""


def page(source: str) -> str:
    return f'<!doctype html><html><body style="margin:0;background:transparent"><pre class="mermaid">{html.escape(source)}</pre></body></html>'


def main() -> None:
    config = json.loads((HERE / "config.json").read_text())
    lib = httpx.get(MERMAID, timeout=120, follow_redirects=True).raise_for_status().text
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for src in sorted(HERE.glob("*.mmd")):
            for name, theme in THEMES.items():
                tab = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=2)
                tab.set_content(page(src.read_text()))
                tab.add_script_tag(content=lib)
                status = tab.evaluate(RUN, [theme, config])
                if status != "OK":
                    raise SystemExit(f"{src.name} ({name}): {status}")
                out = HERE / f"{src.stem}-{name}.png"
                tab.locator("pre.mermaid svg").screenshot(path=str(out), omit_background=True)
                print(f"wrote {out.relative_to(HERE.parent.parent)}")
                tab.close()
        browser.close()


if __name__ == "__main__":
    main()
