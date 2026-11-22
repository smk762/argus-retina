"""``python -m argus_retina.healthcheck``: the image's HEALTHCHECK.

Healthy once the model is loaded, and while a GPU image's model is unloaded (argus-iris freed its VRAM; the next request
loads it again). Unhealthy while it loads, and when it failed to.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    port = os.environ.get("PORT", "8080")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            body = json.load(resp)
    except (OSError, ValueError):
        return 1
    state = body.get("model_state") if isinstance(body, dict) else None
    if state == "unloaded":
        return 0 if body.get("device") == "gpu" else 1
    return 0 if state in ("cpu", "gpu") else 1


if __name__ == "__main__":
    sys.exit(main())
