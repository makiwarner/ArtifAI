import os
import sys
import json
import argparse
import wikipediaapi
import requests
from wikidata.client import Client

# Directory for artist JSONs (relative to script)
ARTISTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'artists')
)

# Initialize clients
wiki = wikipediaapi.Wikipedia(
    user_agent='ArtifAI/1.0 (https://github.com/yourusername/ArtifAI)',
    language='en'
)
wd_client = Client()

# Wikiquote API endpoint
WIKIQUOTE_API = 'https://en.wikiquote.org/w/api.php'


def slugify(name):
    """Convert artist name to filename slug."""
    return name.lower().replace(' ', '_').replace('-', '_')


def fetch_wikipedia_bio(name, paras=2):
    """
    Fetch the first N paragraphs of the Wikipedia page summary as bio.
    """
    page = wiki.page(name)
    if not page.exists():
        print(f"[WARN] Wikipedia page for '{name}' not found.")
        return ""
    # Split summary by newline and join specified paras
    lines = page.summary.split('\n')
    return '\n\n'.join(lines[:paras])


def fetch_wikipedia_sections(name):
    """
    Recursively collect Wikipedia sections and their text into a dict.
    """
    def recurse(sec):
        text = sec.text.strip()
        data = {sec.title: text} if text else {}
        for sub in sec.sections:
            data.update(recurse(sub))
        return data

    page = wiki.page(name)
    if not page.exists():
        return {}
    sections = {}
    for s in page.sections:
        sections.update(recurse(s))
    return sections


def fetch_wikidata_entity(name):
    """Search Wikidata for the entity matching the artist name."""
    try:
        results = wd_client.search(name)
        if not results:
            print(f"[WARN] No Wikidata entity found for '{name}'.")
            return None
        qid = results[0]
        return wd_client.get(qid, load=True)
    except Exception as e:
        print(f"[ERROR] Wikidata search error for '{name}': {e}")
        return None


def fetch_wikidata_fields(entity):
    """
    Extract key properties from a Wikidata entity.
    """
    props = {}
    mapping = [
        ('P569', 'birth_date'),
        ('P570', 'death_date'),
        ('P27', 'nationality'),
        ('P135', 'movement'),
    ]
    for pid, key in mapping:
        try:
            val = entity.get(pid)
            if val:
                if pid in ['P569', 'P570']:
                    props[key] = val.to_time().isoformat()
                else:
                    props[key] = val.label
        except KeyError:
            continue
    # Collect notable works (P800)
    try:
        works = entity.get('P800')
        if works:
            works_list = works if isinstance(works, list) else [works]
            props['notable_works'] = [w.label for w in works_list]
    except KeyError:
        pass
    return props


def fetch_wikiquote_quotes(name, max_quotes=5):
    """
    Fetch top quotes for the artist from Wikiquote.
    """
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'titles': name,
        'explaintext': 1,
        'redirects': 1
    }
    try:
        r = requests.get(WIKIQUOTE_API, params=params, timeout=5)
        data = r.json()
        pages = data.get('query', {}).get('pages', {})
        for pg in pages.values():
            text = pg.get('extract', '')
            # Naively split by line and pick lines that look like quotes
            lines = [l.strip() for l in text.split('\n') if l.strip().startswith('"')]
            return lines[:max_quotes]
    except Exception:
        pass
    return []


def generate_json(name):
    """Generate the JSON structure for a given artist name."""
    # Core schema
    data = {
        'bio': fetch_wikipedia_bio(name),
        'artworks': [],
        'lore': [],
        'knowledge': {}
    }

    # Wikipedia detailed sections
    secs = fetch_wikipedia_sections(name)
    if secs:
        data['knowledge']['sections'] = secs

    # Wikidata enrichment
    entity = fetch_wikidata_entity(name)
    if entity:
        wd = fetch_wikidata_fields(entity)
        # Merge into knowledge
        data['knowledge'].update({
            k: wd.get(k, '') for k in ['birth_date', 'death_date', 'nationality', 'movement']
        })
        data['artworks'] = wd.get('notable_works', [])

    # Wikiquote lore/quotes
    quotes = fetch_wikiquote_quotes(name)
    if quotes:
        data['lore'] = quotes

    return data


def save_json(name, data):
    """Save the JSON data to artists/<slug>.json"""
    os.makedirs(ARTISTS_DIR, exist_ok=True)
    filename = slugify(name) + '.json'
    path = os.path.join(ARTISTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved JSON for '{name}' -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate or regenerate ArtifAI artist JSON files.'
    )
    parser.add_argument(
        'artists', nargs='*', help='Optional artist names; omit to process all.'
    )
    args = parser.parse_args()

    if args.artists:
        names = args.artists
    else:
        # Derive from existing JSON files
        try:
            files = os.listdir(ARTISTS_DIR)
        except FileNotFoundError:
            files = []
        names = [f[:-5].replace('_', ' ') for f in files if f.lower().endswith('.json')]
        if not names:
            print(f"[ERROR] No artist names specified and no JSONs in {ARTISTS_DIR}.")
            sys.exit(1)

    for name in names:
        print(f"Processing '{name}'...")
        data = generate_json(name)
        save_json(name, data)


if __name__ == '__main__':
    main()
