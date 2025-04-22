import pandas as pd
import wikipediaapi
import os

DATA_PATH = "data/artists.csv"
OUT_PATH = "data/best-artworks-updated.csv"

TARGET_ARTISTS = {
    "Sandro Botticelli": "Sandro Botticelli",
    "Leonardo da Vinci": "Leonardo da Vinci",
    "Frida Kahlo": "Frida Kahlo",
    "Gustav Klimt": "Gustav Klimt",
    "Claude Monet": "Claude Monet",
    "Pablo Picasso": "Pablo Picasso",
    "Diego Rivera": "Diego Rivera",
    "Vincent van Gogh": "Vincent van Gogh",
    "Johannes Vermeer": "Johannes Vermeer"
}

def scrape_wikipedia_summary(name):
    wiki = wikipediaapi.Wikipedia(
    user_agent='ArtifAI/1.0 (https://github.com/yourusername/ArtifAI)',
    language='en'
)
    page = wiki.page(name)
    if page.exists():
        return page.summary
    else:
        print(f" Wikipedia page not found for: {name}")
        return None

def build_bio_map():
    bios = {}
    for csv_name, wiki_name in TARGET_ARTISTS.items():
        summary = scrape_wikipedia_summary(wiki_name)
        if summary:
            bios[csv_name] = summary
    return bios

def update_kaggle_bios():
    df = pd.read_csv(DATA_PATH)
    bios = build_bio_map()

    def update_bio(row):
        artist = row['name'] 
        return bios.get(artist, row['bio'])

    df['bio'] = df.apply(update_bio, axis=1)
    df.to_csv(OUT_PATH, index=False)
    print(f"Updated CSV saved to {OUT_PATH}")

if __name__ == "__main__":
    update_kaggle_bios()
