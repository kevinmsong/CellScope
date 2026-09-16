"""Independent DAPI nuclear segmentation and mask-contact counts.

Counts are segmentation estimates, not inferred from cytoplasmic cells. Direct
contact uses 8-neighbour pixel adjacency, including diagonal contacts.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from .clustering import assign_clusters, build_cell_contact_graph
from .image_io import extract_channel
from .types import utc_now


def segment_nuclei(session, channel="blue", min_area=25, seed_distance=5, sigma=1.0,
                   contact_gap=0, source="Image channel"):
    image = session.nuclear_image if source == "Separate nuclear file" else session.image
    if image is None:
        raise ValueError("Load the selected nuclear source first.")
    if min_area < 1 or seed_distance < 1 or sigma < 0 or contact_gap < 0:
        raise ValueError("Nuclear area/seed distance must be positive; smoothing/gap nonnegative.")
    signal = gaussian(extract_channel(image, channel), sigma=float(sigma), preserve_range=True)
    if not np.isfinite(signal).all():
        raise ValueError("Nuclear image contains non-finite intensities.")
    if signal.max() <= signal.min():
        labels = np.zeros(signal.shape, np.int32)
    else:
        mask = signal > threshold_otsu(signal)
        mask = remove_small_objects(mask, min_size=int(min_area))
        distance = ndi.distance_transform_edt(mask)
        coordinates = peak_local_max(distance, min_distance=int(seed_distance), labels=mask,
                                     exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        for i, (y, x) in enumerate(coordinates, 1):
            markers[y, x] = i
        # Ensure every connected foreground component has a seed.
        components, n = ndi.label(mask)
        next_id = len(coordinates) + 1
        for component in range(1, n+1):
            region = components == component
            if not markers[region].any():
                y, x = np.unravel_index(np.argmax(np.where(region, distance, -1)), mask.shape)
                markers[y, x] = next_id
                next_id += 1
        labels = watershed(-distance, markers, mask=mask).astype(np.int32)
        for value, area in zip(*np.unique(labels, return_counts=True)):
            if value and area < min_area:
                labels[labels == value] = 0
    session.nuclei_labels = labels
    session.nuclei_excluded = set(map(int, np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]])))) - {0}
    session.nuclei_settings = dict(channel=channel, min_area=int(min_area), seed_distance=int(seed_distance),
        sigma=float(sigma), contact_gap=int(contact_gap), source=source,
        source_sha256=image.sha256, method="Otsu + distance watershed", direct_contact="8-neighbour adjacency",
        segmented_at=utc_now())
    session.nuclei_log = [{"action": "automatic edge exclusion", "nucleus_id": i, "timestamp": utc_now()}
                          for i in sorted(session.nuclei_excluded)]
    return nuclear_counts(session)


def nuclear_edges(labels, gap=0):
    if gap:
        # Pixel-center radius = one pixel contact + requested background gap.
        return build_cell_contact_graph(labels, int(gap) + 1)
    pairs = set()
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = labels[:labels.shape[0]-dy, max(0, -dx):labels.shape[1]-max(0, dx)]
        b = labels[dy:, max(0, dx):labels.shape[1]-max(0, -dx)]
        good = (a > 0) & (b > 0) & (a != b)
        for x, y in zip(a[good], b[good]):
            pairs.add(tuple(sorted((int(x), int(y)))))
    return sorted(pairs)


def nuclear_tables(session):
    if session.nuclei_labels is None:
        return pd.DataFrame(columns=["nucleus_id", "included", "touching", "group_id", "area_px2"]), []
    raw = session.nuclei_labels
    approved = raw.copy()
    approved[np.isin(approved, list(session.nuclei_excluded))] = 0
    edges = nuclear_edges(approved, session.nuclei_settings.get("contact_gap", 0))
    ids = sorted(set(map(int, np.unique(approved))) - {0})
    groups = assign_clusters(ids, edges)
    touching = {oid for pair in edges for oid in pair}
    rows = [{"nucleus_id": int(oid), "included": oid not in session.nuclei_excluded,
             "touching": oid in touching, "group_id": groups.get(int(oid)), "area_px2": int(area)}
            for oid, area in zip(*np.unique(raw, return_counts=True)) if oid]
    return pd.DataFrame(rows, columns=["nucleus_id", "included", "touching", "group_id", "area_px2"]), edges


def nuclear_counts(session):
    table, edges = nuclear_tables(session)
    accepted = table[table.included.astype(bool)]
    touching = accepted[accepted.touching.astype(bool)]
    return {"nuclei_detected": len(table), "nuclei_included": len(accepted),
            "nuclei_excluded": len(table)-len(accepted), "touching_nuclei": len(touching),
            "isolated_nuclei": len(accepted)-len(touching),
            "touching_groups": int(touching.group_id.nunique()), "contact_pairs": len(edges)}


def batch_nuclear_counts(batch):
    return pd.DataFrame([{"image": s.display_name, "well": s.well, "condition": s.condition,
                          "replicate": s.replicate, **nuclear_counts(s)}
                         for s in batch.images if s.nuclei_labels is not None])


def write_nuclear_files(session, directory):
    import json
    from pathlib import Path

    from .export import write_mask
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_mask(str(directory / "nuclei_labels.tif"), session.nuclei_labels)
    nuclear_tables(session)[0].to_csv(directory / "nuclei_measurements.csv", index=False)
    (directory / "nuclei_metadata.json").write_text(json.dumps({
        "image": session.display_name, "analysis_id": session.analysis_id,
        "settings": session.nuclei_settings, "counts": nuclear_counts(session),
        "excluded_ids": sorted(session.nuclei_excluded), "log": session.nuclei_log}, indent=2), encoding="utf-8")


def export_nuclei(batch, directory):
    import tempfile
    import zipfile
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not any(s.nuclei_labels is not None for s in batch.images):
        raise ValueError("Run DAPI analysis first.")
    with tempfile.TemporaryDirectory(prefix="cellscope_nuclei_") as temporary:
        staging = Path(temporary)
        batch_nuclear_counts(batch).to_csv(staging / "nuclei_counts.csv", index=False)
        for index, session in enumerate(batch.images, 1):
            if session.nuclei_labels is not None:
                write_nuclear_files(session, staging / f"image_{index}")
        path = directory / "nuclei_analysis.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for member in staging.rglob("*"):
                if member.is_file():
                    archive.write(member, member.relative_to(staging).as_posix())
    return str(path)
