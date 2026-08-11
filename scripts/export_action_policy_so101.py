#!/usr/bin/env python3
"""Export SO-101 action-policy DCP -> HF, with PackingDataLoader metadata support.

Upstream ``export_model._build_edge_policy_metadata`` only understands the
``dataloader_train.dataloaders.action_data...`` layout (LIBERO/DROID). Our
``action_policy_so101_edge`` uses PackingDataLoader + ``datasets.so101``.
This wrapper patches that helper, then delegates to export_model.
"""

from __future__ import annotations

import sys
from typing import Any


def _build_so101_edge_policy_metadata(training_config: Any) -> dict[str, Any]:
    """Resolve policy manifest from PackingDataLoader SO101 experiment config."""
    # Preferred: Hydra-instantiated PackingDataLoader layout.
    try:
        ds = training_config.dataloader_train.dataloader.datasets.so101.dataset
        chunk = ds.get("chunk_length") if hasattr(ds, "get") else getattr(ds, "chunk_length", None)
        fps = ds.get("fps") if hasattr(ds, "get") else getattr(ds, "fps", None)
        if chunk is None or fps is None:
            raise AttributeError("chunk_length/fps missing on so101 dataset config")
        return {
            "action_chunk_size": int(chunk),
            "conditioning_fps": float(fps),
            "domain_name": "so101_follower",
        }
    except Exception:
        # Hard fallback matching action_policy_so101_edge defaults.
        return {
            "action_chunk_size": 32,
            "conditioning_fps": 30.0,
            "domain_name": "so101_follower",
        }


def main(argv: list[str] | None = None) -> None:
    # Patch before export_model.main parses / runs.
    import cosmos_framework.scripts.export_model as export_model

    export_model._build_edge_policy_metadata = _build_so101_edge_policy_metadata  # type: ignore[attr-defined]
    sys.argv = [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
    export_model.main()


if __name__ == "__main__":
    main()
