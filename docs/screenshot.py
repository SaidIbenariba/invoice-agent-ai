"""Screenshots of the review screen for the README.

    uv run streamlit run app.py --server.headless true -- --set heldout &
    uv run python docs/screenshot.py inv_007 docs/screen-hold.png
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"


def main(doc_id: str, out: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        page.goto(URL)
        page.wait_for_selector("text=Invoice Agent AI", timeout=60000)
        page.get_by_text(doc_id, exact=False).first.click()
        page.wait_for_timeout(1500)
        page.wait_for_selector("text=Agent trail", timeout=60000)
        page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60000)
        page.wait_for_timeout(2500)
        page.screenshot(path=out, full_page=False)
        browser.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
