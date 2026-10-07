"""Copy approved AD Knowledge Portal files from Synapse to SEA-AD highly-sensitive S3.

Reads `transfers.yaml`, resolves every Synapse file, and for each one downloads to a /scratch staging
directory, checks the MD5 and size against the Synapse file handle, uploads with `aws s3 cp`, checks
the uploaded size, and deletes the staged copy. Files already on S3 with a matching size and MD5
metadata are skipped, so reruns are safe.

Dry run by default; pass --execute to transfer. Credentials are never handled here: Synapse reads
~/.synapseConfig (from `synapse login --remember-me`) and AWS uses the profile named in the config.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import synapseclient
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "transfers.yaml"
DEFAULT_REGISTRY = HERE.parent / "registry" / "datasets.yaml"
DEFAULT_STAGING = Path("/scratch/bulk-comp/staging")


@dataclass
class Item:
    dataset: str
    subdir: str
    syn_id: str
    version: int
    name: str
    size: int
    md5: str
    key: str

    @property
    def s3_uri(self) -> str:
        return f"s3://{BUCKET}/{self.key}"


BUCKET = ""
AWS_ENV: dict[str, str] = {}


def aws(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", *args], env=AWS_ENV, capture_output=True, text=True, check=check)


def head_object(key: str) -> dict | None:
    r = aws("s3api", "head-object", "--bucket", BUCKET, "--key", key, check=False)
    if r.returncode != 0:
        if "Not Found" in r.stderr or "404" in r.stderr:
            return None
        sys.exit(f"head-object failed for {key}: {r.stderr.strip()}")
    return json.loads(r.stdout)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(syn: synapseclient.Synapse, ds: dict, prefix: str) -> list[Item]:
    targets = [(sid, sub) for sub, ids in ds.get("files", {}).items() for sid in ids]
    for spec in ds.get("folders", []):
        for child in syn.getChildren(spec["folder"], includeTypes=["file"]):
            if any(child["name"].endswith(s) for s in spec["suffixes"]):
                targets.append((child["id"], spec["subdir"]))
    items = []
    for sid, sub in targets:
        ent = syn.get(sid, downloadFile=False)
        fh = ent._file_handle
        items.append(Item(ds["name"], sub, ent.id, ent.versionNumber, ent.name, int(fh["contentSize"]),
                          fh["contentMd5"], f"{prefix}/{ds['name']}/{sub}/{ent.name}"))
    return sorted(items, key=lambda i: (i.subdir, i.name))


def on_s3(item: Item) -> bool:
    h = head_object(item.key)
    return h is not None and h["ContentLength"] == item.size and h.get("Metadata", {}).get("md5") == item.md5


def upload(path: Path, key: str, metadata: dict[str, str]) -> None:
    meta = ",".join(f"{k}={v}" for k, v in metadata.items())
    aws("s3", "cp", str(path), f"s3://{BUCKET}/{key}", "--metadata", meta, "--only-show-errors")


def transfer(syn: synapseclient.Synapse, item: Item, staging: Path) -> None:
    dest = staging / item.dataset / item.subdir
    dest.mkdir(parents=True, exist_ok=True)
    ent = syn.get(item.syn_id, version=item.version, downloadLocation=str(dest), ifcollision="overwrite.local")
    path = Path(ent.path)
    size, md5 = path.stat().st_size, md5sum(path)
    if size != item.size or md5 != item.md5:
        sys.exit(f"integrity mismatch for {item.syn_id} {item.name}: size {size}/{item.size} md5 {md5}/{item.md5}")
    upload(path, item.key, {"md5": item.md5, "synapse_id": item.syn_id, "synapse_version": str(item.version)})
    h = head_object(item.key)
    if h is None or h["ContentLength"] != item.size:
        sys.exit(f"S3 size check failed for {item.key}")
    path.unlink()


def git_commit() -> str:
    r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", str(HERE)],
                           capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("-dirty" if dirty else "")


def write_provenance(syn: synapseclient.Synapse, ds: dict, items: list[Item], cfg: dict, staging: Path) -> None:
    """Manifest and README go to S3 beside the data; annotations carry donor IDs, so they stay off git."""
    out = staging / ds["name"]
    out.mkdir(parents=True, exist_ok=True)
    rows = ["\t".join(["synapse_id", "version", "name", "subdir", "s3_uri", "size", "md5", "annotations"])]
    for i in items:
        ann = {k: v for k, v in syn.get_annotations(i.syn_id).items()}
        rows.append("\t".join([i.syn_id, str(i.version), i.name, i.subdir, i.s3_uri, str(i.size), i.md5,
                               json.dumps(ann, default=str)]))
    manifest = out / "SYNAPSE_METADATA_MANIFEST.tsv"
    manifest.write_text("\n".join(rows) + "\n")
    counts = {s: sum(i.subdir == s for i in items) for s in sorted({i.subdir for i in items})}
    readme = out / "README.md"
    readme.write_text(
        f"# {ds['name']}\n\n"
        f"- **Source:** {ds['source']}\n"
        f"- **Access:** {cfg['access']}\n"
        f"- **Copied:** {date.today().isoformat()} from Synapse; every file MD5-verified against its Synapse file handle\n"
        f"- **Files:** {len(items)} ({', '.join(f'{k}: {v}' for k, v in counts.items())}); "
        f"{sum(i.size for i in items) / 1e9:.2f} GB\n"
        f"- **Manifest:** `metadata/SYNAPSE_METADATA_MANIFEST.tsv` (Synapse ID, version, MD5, annotations)\n\n"
        f"_Generated by code/bulk_composition/provision/01_pull_adkp_bulk.py @ {git_commit()}._\n"
    )
    prefix = f"{cfg['prefix']}/{ds['name']}"
    upload(manifest, f"{prefix}/metadata/SYNAPSE_METADATA_MANIFEST.tsv", {"generated_by": "01_pull_adkp_bulk.py"})
    upload(readme, f"{prefix}/README.md", {"generated_by": "01_pull_adkp_bulk.py"})
    manifest.unlink()
    readme.unlink()


def update_registry(path: Path, ds: dict, items: list[Item], cfg: dict) -> None:
    reg = yaml.safe_load(path.read_text()) if path.exists() else {"datasets": {}}
    reg["datasets"][ds["name"]] = {
        "tier": ds["tier"],
        "source": ds["source"],
        "access": cfg["access"],
        "s3_prefix": f"s3://{BUCKET}/{cfg['prefix']}/{ds['name']}/",
        "copied": date.today().isoformat(),
        "n_files": len(items),
        "total_bytes": sum(i.size for i in items),
        "files": [{k: v for k, v in asdict(i).items() if k not in ("dataset", "key")} | {"s3_uri": i.s3_uri}
                  for i in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(reg, sort_keys=False, width=200))


def main() -> None:
    global BUCKET, AWS_ENV
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dataset", action="append", help="dataset name(s) to process; default all, in config order")
    ap.add_argument("--execute", action="store_true", help="transfer; without it only the plan is printed")
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    BUCKET = cfg["bucket"]
    AWS_ENV = os.environ | {"AWS_PROFILE": cfg["aws_profile"]}
    datasets = [d for d in cfg["datasets"] if not args.dataset or d["name"] in args.dataset]
    if args.dataset and len(datasets) != len(args.dataset):
        sys.exit(f"unknown dataset in {args.dataset}")

    syn = synapseclient.Synapse(silent=True)
    syn.login()

    for ds in datasets:
        items = resolve(syn, ds, cfg["prefix"])
        todo = [i for i in items if not on_s3(i)]
        print(f"\n== {ds['tier']} {ds['name']}: {len(items)} files, {sum(i.size for i in items) / 1e9:.2f} GB; "
              f"{len(items) - len(todo)} already on S3, {len(todo)} to copy "
              f"({sum(i.size for i in todo) / 1e9:.2f} GB)")
        for i in items:
            print(f"   {'copy' if i in todo else 'skip'}  {i.syn_id}.{i.version}  {i.size / 1e6:9.1f} MB  {i.s3_uri}")
        if not args.execute:
            continue
        for n, i in enumerate(todo, 1):
            print(f"   [{n}/{len(todo)}] {i.name}", flush=True)
            transfer(syn, i, args.staging)
        write_provenance(syn, ds, items, cfg, args.staging)
        update_registry(args.registry, ds, items, cfg)
        print(f"   done: {ds['name']} verified on S3; registry updated", flush=True)

    if not args.execute:
        print("\nDry run only; rerun with --execute to transfer.")


if __name__ == "__main__":
    main()
