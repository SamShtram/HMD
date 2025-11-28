import trafilatura
import requests
import os
from urllib.parse import urljoin
from bs4 import BeautifulSoup

BASE_SITES = {
    "cdc": "https://www.cdc.gov",
    "who": "https://www.who.int",
    "nih": "https://www.nih.gov",
    "jhu": "https://publichealth.jhu.edu",
    "mayo": "https://www.mayoclinic.org",
    "yale": "https://www.yalemedicine.org",
    "nature": "https://www.nature.com",
    "science": "https://www.science.org"
}

MAX_ARTICLES_PER_SITE = 30

OUT_DIR = "data/credible"
os.makedirs(OUT_DIR, exist_ok=True)

def get_links(base_url):
    try:
        r = requests.get(base_url, timeout=10)
        if r.status_code != 200:
            return []
    except:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = urljoin(base_url, href)
        if base_url in href and len(href) < 200 and href.startswith("http"):
            links.add(href)

    return list(links)

def extract(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        return text
    except:
        return None

def scrape_site(label, base_url):
    links = get_links(base_url)
    count = 0

    for idx, url in enumerate(links):
        if count >= MAX_ARTICLES_PER_SITE:
            break

        text = extract(url)
        if text and len(text) > 800:
            filename = f"{OUT_DIR}/{label}_{count}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            count += 1

def main():
    for label, url in BASE_SITES.items():
        scrape_site(label, url)

if __name__ == "__main__":
    main()
