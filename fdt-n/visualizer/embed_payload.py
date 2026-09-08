"""Embed dashboard-data.json into dashboard.html as the __PAYLOAD__ and sanity-check the result."""

import json
from pathlib import Path

VIS = Path(r"D:\Github\DaxAlgo-Research\fdt-n\visualizer")
html = (VIS / "dashboard.html").read_text(encoding="utf-8")
payload = (VIS / "dashboard-data.json").read_text(encoding="utf-8")

assert "</script" not in payload.lower().replace("<\\/script", ""), "payload contains script terminator"
assert html.count("__PAYLOAD__") == 1
html = html.replace("__PAYLOAD__", payload)
out = VIS / "fdt-n-visualizer.html"
out.write_text(html, encoding="utf-8")
print("written:", out)
print("size MB:", round(out.stat().st_size / 1e6, 2))

# sanity: JSON parses from the embedded slot
start = html.index('<script id="payload" type="application/json">') + len('<script id="payload" type="application/json">')
end = html.index("</script>", start)
data = json.loads(html[start:end])
print("embedded payload keys:", sorted(data.keys()))
print("meta windows_total:", data["meta"]["windows_total"],
      "| replay windows:", len(data["replay"]["timestamps"]),
      "| stages keys:", len(data["stages"]))
