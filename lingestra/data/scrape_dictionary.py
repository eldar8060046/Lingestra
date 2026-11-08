"""Scrape the UAE Sign Language Dictionary (za.gov.ae) into a labelled video dataset.

Uses the site's own public SXA search JSON endpoint (the same one the page calls),
downloads each entry's video and writes data/manifest.json with the English title
(label) and category of every clip. Polite: 1 request/second, resumable.

Usage:  python -m lingestra.data.scrape_dictionary [--index-only] [--limit N]
"""
import argparse, json, re, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_videos"
MANIFEST = ROOT / "data" / "manifest.json"
SEARCH = "https://www.za.gov.ae/en/sxa/search/results/"
PARAMS = dict(sc_lang="en", s="{9D475605-DBA6-4E8A-BA9A-7949E0F32CF5}",
              itemid="{5596362A-FE9B-4ABC-844F-921767C0AE1F}", sig="mastercard",
              v="{FF1E6744-6C43-40A7-835E-D75CE4092535}", p=100, o="Title,Ascending")
HEADERS = {"User-Agent": "Mozilla/5.0 (Lingestra university research; contact: team Vertex)"}
DELAY = 1.0


def fetch_index():
    items, offset = [], 0
    while True:
        r = requests.get(SEARCH, params={**PARAMS, "e": offset}, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not data["Results"]:
            break
        items += data["Results"]
        offset += len(data["Results"])
        if offset >= data["Count"]:
            break
        time.sleep(DELAY)
    return items


def parse(item):
    """Return a manifest entry for English items that carry a video, else None."""
    if item["Language"] != "en":
        return None
    m = re.search(r'href="(https://[^"]+\.mp4[^"]*)"', item["Html"])
    if not m:
        return None
    *_, category, slug = item["Url"].split("/")
    sid = re.search(r"_(\d+)$", slug)
    base = re.sub(r"_\d+$", "", slug)
    variant = re.search(r"-(\d{2})$", base)
    label = re.sub(r"-\d{2}$", "", base).replace("-", " ").strip()
    thumb = re.search(r"url\(&#39;([^&]+)&#39;\)", item["Html"])
    return dict(id=sid.group(1) if sid else item["Id"], category=category.replace("-", " "),
                label=label, variant=int(variant.group(1)) if variant else 1,
                video_url=m.group(1).replace("&amp;", "&"), file=f"{sid.group(1) if sid else item['Id']}.mp4",
                page=item["Url"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    entries = [e for e in map(parse, fetch_index()) if e]
    MANIFEST.write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    print(f"{len(entries)} videos listed -> {MANIFEST}")
    if a.index_only:
        return
    todo = entries[: a.limit] if a.limit else entries
    for i, e in enumerate(todo, 1):
        out = RAW / e["file"]
        if out.exists() and out.stat().st_size > 1000:
            continue
        try:
            r = requests.get(e["video_url"], headers=HEADERS, timeout=120)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("video"):
                out.write_bytes(r.content)
            else:
                print("skip", e["id"], r.status_code, r.headers.get("content-type"))
        except requests.RequestException as ex:
            print("error", e["id"], ex)
        if i % 25 == 0:
            print(f"{i}/{len(todo)}")
        time.sleep(DELAY)


if __name__ == "__main__":
    main()
