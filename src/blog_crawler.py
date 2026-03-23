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

def get_post_links(category_url):
    # Fetch category page
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(category_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        # Find all post titles in WordPress
        for h in soup.find_all(['h1', 'h2'], class_='entry-title'):
            a_tag = h.find('a', href=True)
            if a_tag and a_tag['href'] not in links:
                links.append(a_tag['href'])
        return links
    except Exception as e:
        print(f"Error fetching {category_url}: {e}")
        return []

def scrape_post(url):
    # Fetch individual blog post
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get post title
        title_tag = soup.find('h1', class_='entry-title')
        title = title_tag.text.strip() if title_tag else "Unknown Title"
        
        # Get post content
        content_div = soup.find('div', class_='entry-content')
        if not content_div:
            return ""
            
        # Remove sharing buttons and related posts
        for div in content_div.find_all('div', class_=['sharedaddy', 'jp-relatedposts']):
            div.decompose()
            
        # Extract clean text
        blocks = content_div.find_all(['p', 'h2', 'h3', 'li'])
        lines = [f"# {title}"]

        for block in blocks:
            text = block.get_text(" ", strip=True)
            if _is_stop_heading(text):
                break

            text = re.sub(r'\s+', ' ', text)
            if len(text) <= 5:
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

# Target categories
categories = [
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/12-japan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/22-taiwan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/32-southern-china/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/65-northern-china/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/71-korea/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/75-mongolia/"
]

def main():
    all_docs = []
    print("Starting blog scrape...")

    # Process all categories
    for cat_url in categories:
        print(f"\nScanning category: {cat_url}")
        post_links = get_post_links(cat_url)

        # Scrape each post found in the category
        for i, post_url in enumerate(post_links):
            print(f"  Scraping post [{i+1}/{len(post_links)}]: {post_url}")
            text = scrape_post(post_url)
            if text:
                all_docs.append(text)
            time.sleep(1.0)  # Delay to prevent IP block

    # Save output
    output_dir = "../data" if os.path.exists("../data") else "."
    output_filename = os.path.join(output_dir, "Blog_EastAsian_Cuisines.md")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_docs) + "\n")

    print(f"\nDone! Saved to: {output_filename}")


if __name__ == "__main__":
    main()