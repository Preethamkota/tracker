import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SPLITS = ("train", "validation", "test")


def _source_signature(source_dir):
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [
        [str(path.relative_to(source_dir)), path.stat().st_size]
        for path in files
    ]


def _cache_is_current(source_dir, cache_dir, image_size):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    source_signature = []
    for split in SPLITS:
        source_split = source_dir / split
        source_signature.extend(
            [f"{split}/{relative_path}", size]
            for relative_path, size in _source_signature(source_split)
        )

    return (
        manifest.get("image_size") == image_size
        and manifest.get("source_signature") == source_signature
        and all((cache_dir / split).is_dir() for split in SPLITS)
    )


def preprocess_dataset(data_dir, output_dir, image_size=260, force=False):
    """Convert image-folder splits to cached, resized RGB NumPy arrays."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Image dataset directory not found: {data_dir}")

    source_signature = []
    for split in SPLITS:
        source_split = data_dir / split
        if not source_split.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {source_split}")
        source_signature.extend(
            [f"{split}/{relative_path}", size]
            for relative_path, size in _source_signature(source_split)
        )

    if not force and _cache_is_current(data_dir, output_dir, image_size):
        print(f"Using existing NumPy dataset: {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        source_split = data_dir / split
        cache_split = output_dir / split
        source_dataset = ImageFolder(source_split)
        cache_split.mkdir(parents=True, exist_ok=True)

        for stale_path in cache_split.rglob("*.npy"):
            stale_path.unlink()

        for source_path, _ in source_dataset.samples:
            relative_path = Path(source_path).relative_to(source_split)
            output_path = (
                cache_split
                / relative_path.parent
                / f"{relative_path.stem}{relative_path.suffix}.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source_path) as image:
                image_array = np.asarray(
                    image.convert("L")
                    .resize((image_size, image_size))
                    .convert("RGB"),
                    dtype=np.uint8,
                )

            np.save(output_path, image_array, allow_pickle=False)

    manifest = {
        "image_size": image_size,
        "source_signature": source_signature,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Converted image dataset to NumPy arrays: {output_dir}")
    return output_dir


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    preprocess_dataset(
        data_dir=project_dir / "game_data",
        output_dir=project_dir / "game_data_npy",
    )
