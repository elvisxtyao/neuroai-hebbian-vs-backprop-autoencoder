"""Extract aligned representations without changing encoder state."""

from __future__ import annotations

from collections.abc import Iterable

import torch


@torch.no_grad()
def extract_representations(
    model,
    loader,
    *,
    device: torch.device,
    layers: Iterable[str] = ("h1", "h2", "z"),
) -> dict[str, torch.Tensor]:
    requested = tuple(layers)
    allowed = {"h1", "h2", "z"}
    if not set(requested).issubset(allowed):
        raise ValueError(f"layers must be a subset of {sorted(allowed)}")

    model.eval()
    collected: dict[str, list[torch.Tensor]] = {layer: [] for layer in requested}
    labels: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []
    for batch in loader:
        if len(batch) != 3:
            raise ValueError("Loader must return image, label and stable sample_id")
        images, batch_labels, batch_ids = batch
        features = model.encode(images.to(device), return_all_layers=True)
        for layer in requested:
            collected[layer].append(features[layer].detach().cpu())
        labels.append(torch.as_tensor(batch_labels).cpu())
        sample_ids.append(torch.as_tensor(batch_ids).cpu())

    output = {layer: torch.cat(values) for layer, values in collected.items()}
    output["label"] = torch.cat(labels)
    output["sample_id"] = torch.cat(sample_ids)
    return output

