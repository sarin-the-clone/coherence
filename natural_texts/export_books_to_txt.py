import os
import zipfile
import glob
import shlex
import shutil
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configuration
SOURCE_DIR = 'books'
OUTPUT_DIR = 'Converted_Txt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Clean up extracted text."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return '\n\n'.join(lines)

def extract_metadata_and_spine(zip_ref):
    """Parse OPF to get metadata and reading order (spine)."""
    # Find .opf file
    opf_path = None
    for name in zip_ref.namelist():
        if name.endswith('.opf'):
            opf_path = name
            break
    
    if not opf_path:
        return {}, []

    try:
        with zip_ref.open(opf_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            
            # Namespaces likely used in OPF
            ns = {
                'opf': 'http://www.idpf.org/2007/opf',
                'dc': 'http://purl.org/dc/elements/1.1/'
            }
            # Handle cases where OPF might not use namespaces strictly or uses different ones
            # For simplicity, we search by local name in complex cases, but normally ns works.
            
            metadata = {}
            # Title
            title = root.find('.//dc:title', ns)
            metadata['title'] = title.text.strip() if title is not None else "Unknown Title"
            
            # Author
            creator = root.find('.//dc:creator', ns)
            metadata['author'] = creator.text.strip() if creator is not None else "Unknown Author"
            
            # Spine and manifest
            manifest = {}
            for item in root.findall('.//opf:manifest/opf:item', ns):
                item_id = item.get('id')
                href = item.get('href')
                manifest[item_id] = href
                
            spine = []
            for itemref in root.findall('.//opf:spine/opf:itemref', ns):
                item_id = itemref.get('idref')
                if item_id in manifest:
                    spine.append(manifest[item_id])
            
            # Resolve relative paths relative to OPF file
            opf_dir = os.path.dirname(opf_path)
            full_spine = []
            for href in spine:
                full_path = os.path.join(opf_dir, href).replace('\\', '/')
                # Normalize path (remove ./ etc)
                # Simple normalization (zipfile doesn't like .. usually)
                full_spine.append(full_path)
                
            return metadata, full_spine

    except Exception as e:
        print(f"Error parsing OPF {opf_path}: {e}")
        return {}, []

import textwrap

# ... (imports)

def convert_epub(epub_path):
    try:
        filename = os.path.basename(epub_path)
        print(f"Processing {filename}...")
        
        with zipfile.ZipFile(epub_path, 'r') as z:
            meta, spine = extract_metadata_and_spine(z)
            
            if not spine:
                # Fallback: list all html/xhtml files
                spine = [n for n in z.namelist() if n.endswith(('.html', '.xhtml', '.htm'))]
                spine.sort()
            
            # Sanitize filename for output
            safe_title = "".join([c for c in meta.get('title', 'Unknown') if c.isalpha() or c.isdigit() or c==' ']).strip()
            safe_author = "".join([c for c in meta.get('author', 'Unknown') if c.isalpha() or c.isdigit() or c==' ']).strip()
            if not safe_title: safe_title = "UnknownTitle"
            
            out_filename = f"{safe_title} - {safe_author}.txt".replace('  ', ' ')
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            
            full_content = []
            
            # Header
            full_content.append(f"*** START OF THE PROJECT GUTENBERG EBOOK {meta.get('title', 'UNKNOWN').upper()} ***")
            full_content.append(f"\nTitle: {meta.get('title', 'Unknown')}")
            full_content.append(f"Author: {meta.get('author', 'Unknown')}")
            full_content.append("\n" + "-"*40 + "\n")
            
            wrapper = textwrap.TextWrapper(width=72, break_long_words=False, break_on_hyphens=False)

            for file_path in spine:
                try:
                    # ... (file path normalization)
                    
                    with z.open(file_path) as f:
                        html_content = f.read()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Remove javascript and style
                        for script in soup(["script", "style"]):
                            script.extract()
                            
                        # Get text
                        # We want to preserve paragraph structure but wrap lines.
                        # soup.get_text() merges everything.
                        # Better to iterate over paragraphs if possible, or just split by double newline after get_text
                        
                        # Simple approach: get all text, clean empty lines, then wrap paragraphs
                        text = soup.get_text('\n\n') # Separator between tags
                        
                        # Split into paragraphs
                        paragraphs = text.split('\n\n')
                        for p in paragraphs:
                            cleaned_p = " ".join(p.split()) # Remove internal newlines/spaces
                            if cleaned_p:
                                wrapped = wrapper.fill(cleaned_p)
                                full_content.append(wrapped)
                            
                except KeyError:
                    pass 
                except Exception as e:
                    print(f"Error reading {file_path} in {filename}: {e}")

            full_content.append(f"\n*** END OF THE PROJECT GUTENBERG EBOOK {meta.get('title', 'UNKNOWN').upper()} ***")
            
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(full_content))
                
        return True
    except Exception as e:
        print(f"Failed to convert {epub_path}: {e}")
        return False

def main():
    epub_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith('.epub'):
                epub_files.append(os.path.join(root, file))
    
    print(f"Found {len(epub_files)} EPUB files.")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(convert_epub, epub_files))
        
    print(f"Converted {sum(results)}/{len(epub_files)} books.")

if __name__ == "__main__":
    main()
