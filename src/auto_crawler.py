import requests
from bs4 import BeautifulSoup
import re
import time
import os


def _normalize_heading_key(text):
    return re.sub(r'[^a-z]+', '_', text.strip().lower()).strip('_')


def _is_stop_heading(text):
    stop_headings = {
        'references', 'external_links', 'further_reading', 'see_also', 'notes', 'bibliography'
    }
    return _normalize_heading_key(text) in stop_headings

def get_real_recipes_from_categories(category_urls):
    print("Scanning underlying MediaWiki category pages...")
    all_links = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for url in category_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            category_div = soup.find('div', {'class': 'mw-category'})
            if not category_div:
                continue
                
            for a_tag in category_div.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith('/wiki/Cookbook:'):
                    full_url = "https://en.wikibooks.org" + href
                    if full_url not in all_links:
                        all_links.append(full_url)
                        
        except Exception as e:
            print(f"Error scanning category {url}: {e}")
            
    print(f"Success! Found {len(all_links)} actual recipe links.\n")
    return all_links

def scrape_recipe_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_tag = soup.find('h1')
        title = title_tag.text if title_tag else "Unknown"
        
        content = soup.find('div', {'class': 'mw-parser-output'})
        if not content:
            return ""
            
        for element in content.find_all(['sup', 'span', 'table', 'div']):
            if element.name == 'sup' and 'reference' in element.get('class', []):
                element.decompose()
            elif element.name == 'span' and 'mw-editsection' in element.get('class', []):
                element.decompose()
            elif element.name == 'table' or (element.name == 'div' and 'navbox' in element.get('class', [])):
                element.decompose() 
            
        blocks = content.find_all(['p', 'h2', 'h3', 'li'])

        lines = [f"# {title}"]
        for block in blocks:
            text = block.get_text(" ", strip=True)

            if text.startswith("Cookbook |") or "Cookbook Disambiguation Pages" in text:
                continue
            if "Incomplete recipes" in text or "deletion policy" in text or "meaningful content" in text:
                continue
            if text in {"v", "t", "e"}:
                continue
            if _is_stop_heading(text):
                break

            text = re.sub(r'\s+', ' ', text)
            if len(text) <= 3:
                continue

            if block.name == 'h2':
                lines.append(f"## {text}")
            elif block.name == 'h3':
                lines.append(f"### {text}")
            elif block.name == 'li':
                lines.append(f"* {text}")
            else:
                lines.append(text)

        return "\n\n".join(lines)
    
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

category_urls = [
    "https://en.wikibooks.org/wiki/Category:Chinese_recipes",
    "https://en.wikibooks.org/wiki/Category:Japanese_recipes",
    "https://en.wikibooks.org/wiki/Category:Korean_recipes",
    "https://en.wikibooks.org/wiki/Category:Taiwanese_recipes"
]

def main():
    recipe_urls = get_real_recipes_from_categories(category_urls)

    all_docs = []
    print("Starting bulk recipe extraction...")

    for i, url in enumerate(recipe_urls):
        print(f"[{i+1}/{len(recipe_urls)}] Scraping: {url}")
        scraped_text = scrape_recipe_text(url)
        if scraped_text:
            all_docs.append(scraped_text)
        time.sleep(1.0)

    output_dir = "../data" if os.path.exists("../data") else "."
    output_filename = os.path.join(output_dir, "Wikibooks_EastAsian_Recipes_Clean.md")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_docs) + "\n")

    print(f"\nDone! Recipes saved to: {output_filename}")


if __name__ == "__main__":
    main()