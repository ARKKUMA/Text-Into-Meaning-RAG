import requests
from bs4 import BeautifulSoup, FeatureNotFound
import re
import time
import os
import html2text
from urllib.parse import unquote


def _get_heading_id(heading):
    # Wikipedia heading id can be on the heading tag itself or on a nested mw-headline span.
    direct_id = heading.get("id")
    if direct_id:
        return direct_id

    headline = heading.find('span', {'class': 'mw-headline'})
    if headline:
        return headline.get("id")

    return None


def _extract_target_section(content, soup, section_id):
    # Try both raw and underscore-normalized fragment ids.
    target = content.find(id=section_id) or content.find(id=section_id.replace(" ", "_"))
    if not target:
        return content

    # Target may be a heading itself (newer HTML) or a nested span (older HTML).
    header = target if target.name in ['h2', 'h3', 'h4', 'h5'] else target.find_parent(['h2', 'h3', 'h4', 'h5'])
    if not header:
        return content

    # Newer Wikipedia wraps heading tags in a div.mw-heading container.
    header_block = header
    parent = header.parent
    if parent and parent.name == 'div' and parent.get('class') and 'mw-heading' in parent.get('class'):
        header_block = parent

    html_parts = [str(header_block)]

    for sibling in header_block.find_next_siblings():
        # Stop at the next top heading block (modern HTML) or heading tag (older HTML).
        sibling_classes = sibling.get('class') if hasattr(sibling, 'get') else None
        if sibling.name == 'div' and sibling_classes and 'mw-heading' in sibling_classes:
            break
        if sibling.name in ['h1', 'h2']:
            break
        html_parts.append(str(sibling))

    section_html = f"<div>{''.join(html_parts)}</div>"
    return BeautifulSoup(section_html, 'html.parser').div


def _remove_heading_and_tail(heading):
    # Remove heading block plus all following siblings in the same extraction container.
    heading_block = heading
    parent = heading.parent
    if parent and parent.name == 'div' and parent.get('class') and 'mw-heading' in parent.get('class'):
        heading_block = parent

    for sibling in heading_block.find_next_siblings():
        sibling.decompose()
    heading_block.decompose()


def _normalize_heading_key(text):
    # Normalize heading text/id for robust matching across case/space/underscore differences.
    return re.sub(r'[^a-z]+', '_', text.strip().lower()).strip('_')


def _extract_markdown_anchor_section(markdown_text, section_id):
    section_key = _normalize_heading_key(section_id)
    lines = markdown_text.split('\n')

    in_section = False
    start_level = None
    selected = []

    for line in lines:
        stripped = line.strip()
        heading_match = re.match(r'^(#{1,6})\s*(.+?)\s*$', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            heading_key = _normalize_heading_key(heading_text)

            key_match = heading_key == section_key or re.search(
                rf'(^|_){re.escape(section_key)}($|_)',
                heading_key
            )
            if not in_section and key_match:
                in_section = True
                start_level = level

            elif in_section and level <= start_level:
                break

        if in_section:
            selected.append(line)

    return '\n'.join(selected).strip()


def _is_low_value_url(url):
    # List/index pages are usually navigation-heavy and low value for retrieval quality.
    url_lower = url.lower()

    # Keep this curated section: list page but only East Asian cuisine anchor.
    if "/wiki/list_of_asian_cuisines" in url_lower and "#" in url_lower:
        frag = url_lower.split('#', 1)[1].strip().replace(" ", "_")
        if frag == "east_asian_cuisine":
            return False

    low_value_patterns = [
        "/wiki/List_of_",
        "/wiki/Index_of_",
        "/wiki/Category:",
    ]
    return any(pattern.lower() in url_lower for pattern in low_value_patterns)

def clean_wiki_text(url):
    try:
        # Set user agent to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except FeatureNotFound:
            try:
                soup = BeautifulSoup(response.text, 'html5lib')
            except FeatureNotFound:
                soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get page title
        title_element = soup.find('h1')
        title = title_element.text if title_element else "Unknown Title"
        
        # Get main content area
        full_content = soup.find('div', {'class': 'mw-parser-output'})
        content = full_content
        if not content:
            return ""
        section_id = None

        if '#' in url:
            section_id = unquote(url.split('#', 1)[-1]).strip()
            content = _extract_target_section(content, soup, section_id)

        junk_ids = {'references', 'external_links', 'bibliography', 'notes', 'see_also', 'further_reading'}
        for heading in content.find_all(['h2', 'h3', 'h4']):
            heading_id = _get_heading_id(heading)
            normalized_id = _normalize_heading_key(heading_id) if heading_id else ""
            normalized_text = _normalize_heading_key(heading.get_text(" ", strip=True))
            if normalized_id in junk_ids or normalized_text in junk_ids:
                _remove_heading_and_tail(heading)
                break
        
        # Remove references like [1] and edit links
        for sup in content.find_all('sup', {'class': 'reference'}):
            sup.decompose()
        for span in content.find_all('span', {'class': 'mw-editsection'}):
            span.decompose()

        h = html2text.HTML2Text()
        h.body_width = 0
        h.unicode_snob = True
        h.ignore_links = True   
        h.ignore_images = True  
        
        raw_markdown = h.handle(str(content))

        # Fallback: if anchor extraction is too short, extract by markdown heading range.
        if section_id and len(raw_markdown.strip()) < 200 and full_content:
            full_markdown = h.handle(str(full_content))
            extracted = _extract_markdown_anchor_section(full_markdown, section_id)
            if extracted:
                raw_markdown = extracted
        
        clean_lines = [f"# {title}"]
        
        stop_headings = {'references', 'external_links', 'further_reading', 'see_also', 'notes', 'bibliography'}

        for line in raw_markdown.split('\n'):
            text = line.strip()
            normalized_line = _normalize_heading_key(re.sub(r'^#+\s*', '', text))

            # If a metadata section heading appears, stop consuming the page content.
            if normalized_line in stop_headings:
                break
            
            if text.startswith("Cookbook |") or "Cookbook Disambiguation Pages" in text:
                continue
            if "Incomplete recipes" in text or "deletion policy" in text or "meaningful content" in text:
                continue
            if text in {"References", "External links", "Further reading", "See also", "Notes", "Bibliography", "v", "t", "e"}:
                continue
            if "Retrieved" in text and "Archived" in text:
                continue
                
            is_structure = text.startswith(('#', '*', '-', '1.'))
            if is_structure or len(text) > 2: 
                text = re.sub(r'\s+', ' ', text)
                clean_lines.append(text)
                
        return "\n\n".join(clean_lines)
    
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""
    
# List of Wikipedia and Wikibooks URLs to scrape
urls = [
    "https://en.wikipedia.org/wiki/List_of_Asian_cuisines#East_Asian_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Anhui_cuisine",
    "https://en.wikipedia.org/wiki/Cantonese_cuisine",
    "https://en.wikipedia.org/wiki/Fujian_cuisine",
    "https://en.wikipedia.org/wiki/Hunan_cuisine",
    "https://en.wikipedia.org/wiki/Jiangsu_cuisine",
    "https://en.wikipedia.org/wiki/Shandong_cuisine",
    "https://en.wikipedia.org/wiki/Sichuan_cuisine",
    "https://en.wikipedia.org/wiki/Zhejiang_cuisine",
    "https://en.wikipedia.org/wiki/Dim_sum",
    "https://en.wikipedia.org/wiki/Hot_pot",
    "https://en.wikipedia.org/wiki/Wine_in_China",
    "https://en.wikipedia.org/wiki/Char_siu",
    "https://en.wikipedia.org/wiki/Sichuan_peppercorn",
    "https://en.wikipedia.org/wiki/Huaiyang_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_Islamic_cuisine",
    "https://en.wikipedia.org/wiki/Beijing_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_aristocrat_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_imperial_cuisine",
    "https://en.wikipedia.org/wiki/Liaoning_cuisine",
    "https://en.wikipedia.org/wiki/Chaozhou_cuisine",
    "https://en.wikipedia.org/wiki/Chiuchow_cuisine",
    "https://en.wikipedia.org/wiki/Guizhou_cuisine",
    "https://en.wikipedia.org/wiki/Hainan_cuisine",
    "https://en.wikipedia.org/wiki/Hakka_cuisine",
    "https://en.wikipedia.org/wiki/Henan_cuisine",
    "https://en.wikipedia.org/wiki/Hubei_cuisine",
    "https://en.wikipedia.org/wiki/Jiangxi_cuisine",
    "https://en.wikipedia.org/wiki/Manchu_cuisine",
    "https://en.wikipedia.org/wiki/Northeastern_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Shaanxi_cuisine",
    "https://en.wikipedia.org/wiki/Shanghai_cuisine",
    "https://en.wikipedia.org/wiki/Shanxi_cuisine",
    "https://en.wikipedia.org/wiki/Tianjin_cuisine",
    "https://en.wikipedia.org/wiki/Tibetan_cuisine",
    "https://en.wikipedia.org/wiki/Uyghur_cuisine",
    "https://en.wikipedia.org/wiki/Yunnan_cuisine",
    "https://en.wikipedia.org/wiki/Hong_Kong_cuisine",
    "https://en.wikipedia.org/wiki/Fish_balls",
    "https://en.wikipedia.org/wiki/Wonton_noodle",
    "https://en.wikipedia.org/wiki/Egg_waffle",
    "https://en.wikipedia.org/wiki/Japanese_cuisine",
    "https://en.wikipedia.org/wiki/Japanese_regional_cuisine",
    "https://en.wikipedia.org/wiki/Kaiseki",
    "https://en.wikipedia.org/wiki/Sushi",
    "https://en.wikipedia.org/wiki/Sashimi",
    "https://en.wikipedia.org/wiki/Japanese_wine",
    "https://en.wikipedia.org/wiki/Okinawan_cuisine",
    "https://en.wikipedia.org/wiki/Awamori",
    "https://en.wikipedia.org/wiki/Nagoya_cuisine",
    "https://en.wikipedia.org/wiki/Ainu_cuisine",
    "https://en.wikipedia.org/wiki/Korean_cuisine",
    "https://en.wikipedia.org/wiki/Banchan",
    "https://en.wikipedia.org/wiki/Kimchi",
    "https://en.wikipedia.org/wiki/Doenjang",
    "https://en.wikipedia.org/wiki/Korean_soy_sauce",
    "https://en.wikipedia.org/wiki/Gochujang",
    "https://en.wikipedia.org/wiki/Korean_regional_cuisine",
    "https://en.wikipedia.org/wiki/Korean_barbecue",
    "https://en.wikipedia.org/wiki/Soju",
    "https://en.wikipedia.org/wiki/Makgeolli",
    "https://en.wikipedia.org/wiki/Korean_royal_court_cuisine",
    "https://en.wikipedia.org/wiki/Korean_temple_cuisine",
    "https://en.wikipedia.org/wiki/North_Korean_cuisine",
    "https://en.wikipedia.org/wiki/South_Korean_cuisine",
    "https://en.wikipedia.org/wiki/Mongolian_cuisine",
    "https://en.wikipedia.org/wiki/Taiwanese_cuisine",
    "https://en.wikipedia.org/wiki/Khorkhog",
    "https://en.wikipedia.org/wiki/Japanese_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Shippoku",
    "https://en.wikipedia.org/wiki/Itameshi",
    "https://en.wikipedia.org/wiki/Yoshoku",
    "https://en.wikipedia.org/wiki/Korean_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Hot_and_sour_noodles",
    "https://en.wikipedia.org/wiki/Xiaolongbao",
    "https://en.wikipedia.org/wiki/Xuzhou_cuisine",
    "https://en.wikipedia.org/wiki/Haipai_cuisine",
    "https://en.wikipedia.org/wiki/Qinghai_cuisine",
    "https://en.wikipedia.org/wiki/Guilin_cuisine",
    "https://en.wikipedia.org/wiki/Putian_cuisine",
    "https://en.wikipedia.org/wiki/Teochew_cuisine",
    "https://en.wikipedia.org/wiki/Ou_cuisine",
    "https://en.wikipedia.org/wiki/Kachin_cuisine",
    "https://en.wikipedia.org/wiki/Hmong_cuisine",
    "https://en.wikipedia.org/wiki/Taoist_diet",
    "https://en.wikipedia.org/wiki/History_of_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/History_of_Japanese_cuisine",
    "https://en.wikibooks.org/wiki/Cookbook:Broccoli_Stir_Fry",
    "https://en.wikibooks.org/wiki/Cookbook:Cream_Cheese_Wontons",
    "https://en.wikibooks.org/wiki/Cookbook:Fried_Rice",
    "https://en.wikibooks.org/wiki/Cookbook:Onigiri",
    "https://en.wikibooks.org/wiki/Cookbook:Spicy_Miso_Udon",
    "https://en.wikibooks.org/wiki/Cookbook:Wonton_Soup"
]

def main():
    all_text = ""
    print("Starting to scrape East Asian cuisine corpus...")

    for url in urls:
        if _is_low_value_url(url):
            print(f"Skipping low-value page: {url}")
            continue

        print(f"Fetching: {url}")
        scraped_text = clean_wiki_text(url)
        if scraped_text:
            all_text += scraped_text + "\n\n---\n\n"
        # Delay to prevent IP blocking
        time.sleep(1.5)

    # Save to data directory if it exists, otherwise use current directory
    output_dir = "../data" if os.path.exists("../data") else "."
    output_filename = os.path.join(output_dir, "East_Asian_Corpus_Massive.md")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"\nDone! Saved to: {output_filename}")
    print(f"Total characters: {len(all_text)}")


if __name__ == "__main__":
    main()