#!/usr/bin/env python3
"""
Download free NVIDIA Omniverse USD asset packs.

These packs are freely downloadable — no account or API key required.
Assets are stored in ~/.cache/scenesmith/assets/ and automatically
discovered by the SceneSmith MCP server.

Usage:
    python scripts/download_assets.py                   # Download recommended packs
    python scripts/download_assets.py --pack furniture  # Download a specific pack
    python scripts/download_assets.py --list            # List available packs
    python scripts/download_assets.py --dry-run         # Show what would be downloaded

Available packs:
    furniture   SimReady Furniture & Misc (9.4 GB, 202 models) ← recommended
    residential Residential Assets (22.5 GB, 507 items)
    materials   Base Materials (8.2 GB, 161 materials)
    commercial  Commercial Assets (5.8 GB, 82 pieces)
    scenes      Sample Scenes (26 GB, 441 assets)

License: NVIDIA Omniverse License Agreement
  https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# ── Asset pack catalog ─────────────────────────────────────────────────────────
PACKS = {
    "furniture": {
        "name": "SimReady Furniture & Misc",
        "description": "202 physics-ready furniture models: chairs, tables, sofas, kitchen items",
        "size_gb": 9.4,
        "url": "https://d4i3qtqj3r0z5.cloudfront.net/SimReady_Furniture_Misc_01_NVD%4010010.zip",
        "extract_subdir": "SimReady_Furniture",
        "recommended": True,
    },
    "residential": {
        "name": "Residential Assets",
        "description": "507 items: furniture, fixtures, decorations, electronics, food",
        "size_gb": 22.5,
        "url": "https://d4i3qtqj3r0z5.cloudfront.net/Residential_NVD%4010012.zip",
        "extract_subdir": "Residential",
        "recommended": False,
    },
    "materials": {
        "name": "Base Materials",
        "description": "161 photoreal materials: carpet, glass, metals, wood, brick, fabric",
        "size_gb": 8.2,
        "url": "https://d4i3qtqj3r0z5.cloudfront.net/Base_Materials_NVD%4010013.zip",
        "extract_subdir": "Base_Materials",
        "recommended": False,
    },
    "commercial": {
        "name": "Commercial Assets",
        "description": "82 pieces: office furniture, desks, tables, chairs",
        "size_gb": 5.8,
        "url": "https://d4i3qtqj3r0z5.cloudfront.net/Commercial_NVD%4010013.zip",
        "extract_subdir": "Commercial",
        "recommended": False,
    },
    "scenes": {
        "name": "Sample Scenes",
        "description": "441 assets in themed scene configurations",
        "size_gb": 26.0,
        "url": "https://d4i3qtqj3r0z5.cloudfront.net/Sample_Scenes_NVD%4010013.zip",
        "extract_subdir": "Sample_Scenes",
        "recommended": False,
    },
}

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "scenesmith" / "assets"
CACHE_DIR = _DEFAULT_CACHE_DIR


def human_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def check_disk_space(required_gb: float, target_dir: Path) -> bool:
    """Check if there's enough disk space."""
    stat = shutil.disk_usage(target_dir.parent if not target_dir.exists() else target_dir)
    available_gb = stat.free / (1024 ** 3)
    if available_gb < required_gb * 1.2:  # 20% buffer
        print(f"  WARNING: Need ~{required_gb:.1f} GB, only {available_gb:.1f} GB available")
        return False
    return True


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a progress bar."""
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = min(count * block_size * 100 / total_size, 100)
            downloaded = human_size(min(count * block_size, total_size))
            total = human_size(total_size)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {percent:.1f}% ({downloaded}/{total})", end="", flush=True)

    print(f"  Downloading to: {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress bar


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file with progress."""
    print(f"  Extracting to: {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members):
            if i % 100 == 0:
                print(f"\r  Extracting... {i}/{total} files", end="", flush=True)
            zf.extract(member, extract_to)
    print(f"\r  Extracted {total} files.          ")


def download_pack(pack_key: str, dry_run: bool = False) -> bool:
    """Download and extract a single pack."""
    if pack_key not in PACKS:
        print(f"Unknown pack: {pack_key}. Run with --list to see available packs.")
        return False

    pack = PACKS[pack_key]
    extract_dir = CACHE_DIR / pack["extract_subdir"]

    print(f"\n{'='*60}")
    print(f"  Pack: {pack['name']}")
    print(f"  Size: ~{pack['size_gb']} GB")
    print(f"  Dest: {extract_dir}")
    print(f"{'='*60}")

    if extract_dir.exists() and any(extract_dir.rglob("*.usd")):
        usd_count = len(list(extract_dir.rglob("*.usd")))
        print(f"  Already downloaded ({usd_count} USD files found). Skipping.")
        return True

    if dry_run:
        print(f"  [DRY RUN] Would download {pack['size_gb']} GB from:")
        print(f"  {pack['url']}")
        return True

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not check_disk_space(pack["size_gb"] * 2, CACHE_DIR):
        response = input("  Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            return False

    # Download to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".zip", dir=CACHE_DIR, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_with_progress(pack["url"], tmp_path)
        extract_zip(tmp_path, extract_dir)
        tmp_path.unlink()

        usd_count = len(list(extract_dir.rglob("*.usd")))
        print(f"  Done! {usd_count} USD files extracted to {extract_dir}")
        return True

    except KeyboardInterrupt:
        print("\n  Download cancelled.")
        if tmp_path.exists():
            tmp_path.unlink()
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def list_packs() -> None:
    """Print available packs."""
    print("\nAvailable NVIDIA Omniverse USD Asset Packs")
    print("(Free download, no account required)")
    print("=" * 60)

    already_downloaded = []
    for key, pack in PACKS.items():
        extract_dir = CACHE_DIR / pack["extract_subdir"]
        is_downloaded = extract_dir.exists() and any(extract_dir.rglob("*.usd"))
        status = "✓ downloaded" if is_downloaded else f"~{pack['size_gb']} GB"
        rec = " [recommended]" if pack.get("recommended") else ""
        print(f"\n  {key}{rec}")
        print(f"    {pack['name']} ({status})")
        print(f"    {pack['description']}")
        if is_downloaded:
            already_downloaded.append(key)

    if already_downloaded:
        print(f"\nAlready downloaded: {', '.join(already_downloaded)}")
    print()


def show_status() -> None:
    """Show current download status."""
    print(f"\nAsset cache: {CACHE_DIR}")
    total_usd = 0
    for key, pack in PACKS.items():
        extract_dir = CACHE_DIR / pack["extract_subdir"]
        if extract_dir.exists():
            usd_files = list(extract_dir.rglob("*.usd"))
            total_usd += len(usd_files)
            print(f"  {pack['name']}: {len(usd_files)} USD files")
    if total_usd == 0:
        print("  No packs downloaded yet.")
    else:
        print(f"  Total: {total_usd} USD files")


def main():
    parser = argparse.ArgumentParser(
        description="Download free NVIDIA Omniverse USD asset packs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pack",
        choices=list(PACKS.keys()),
        help="Download a specific pack (default: furniture)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download ALL packs (warning: ~70 GB total)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available packs and download status",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current download status",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {_DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()

    global CACHE_DIR
    CACHE_DIR = args.cache_dir

    if args.list:
        list_packs()
        return

    if args.status:
        show_status()
        return

    if args.all:
        total_gb = sum(p["size_gb"] for p in PACKS.values())
        print(f"\nDownloading ALL packs (~{total_gb:.0f} GB total)")
        if not args.dry_run:
            confirm = input("This will use a lot of disk space. Continue? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return
        for key in PACKS:
            download_pack(key, dry_run=args.dry_run)
        return

    # Default: download recommended pack (furniture) or specified pack
    pack_key = args.pack or "furniture"
    success = download_pack(pack_key, dry_run=args.dry_run)

    if success and not args.dry_run:
        print("\nAssets are now available to SceneSmith.")
        print("The MCP server will auto-discover them on next start.")
        print(f"Location: {CACHE_DIR / PACKS[pack_key]['extract_subdir']}")


if __name__ == "__main__":
    main()
