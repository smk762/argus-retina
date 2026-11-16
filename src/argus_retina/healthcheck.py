"""``python -m argus_retina.healthcheck``: the image's HEALTHCHECK. Healthy once the model is loaded."""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    port = os.environ.get("PORT", "8080")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            state = json.load(resp).get("model_state")
    except (OSError, ValueError):
        return 1
    return 0 if state in ("cpu", "gpu") else 1


if __name__ == "__main__":
    sys.exit(main())
