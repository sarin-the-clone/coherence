import os
import shutil
import textwrap
import mobi
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import logging

# Configuration
SOURCE_DIR = 'books'
OUTPUT_DIR = 'Converted_Mobi_Txt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_text_wrapper(text, width=72):
    """Clean and hard-wrap text."""
    wrapper = textwrap.TextWrapper(width=width, break_long_words=False, break_on_hyphens=False)
    
    cleaned_paragraphs = []
    # Basic paragraph splitting (assuming double newlines or similar structure)
    # Since BeautifulSoup get_text() can be messy, we rely on semantic tags if possible
    # But mobi extract often gives just one big HTML.
    
    lines = text.splitlines()
    buffer = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                # Flush buffer as a paragraph
                p_text = " ".join(buffer)
                wrapped = wrapper.fill(p_text)
                cleaned_paragraphs.append(wrapped)
                buffer = []
        else:
            buffer.append(line)
    
    if buffer:
        p_text = " ".join(buffer)
        cleaned_paragraphs.append(wrapper.fill(p_text))
        
    return '\n\n'.join(cleaned_paragraphs)

def convert_mobi(mobi_path):
    try:
        filename = os.path.basename(mobi_path)
        logging.info(f"Processing {filename}...")
        
        # Extract MOBI to temp directory
        temp_dir, file_path = mobi.extract(mobi_path)
        
        # Determine Title/Author from filename (Mobi metadata extraction is tricky with just 'mobi' lib)
        # Using filename convention: "Author - Title.mobi" or "Title - Author.mobi"
        # The provided file list shows "Author - Title.mobi" often, or just Titles.
        # We will use the filename as Title for simplicity if parsing fails.
        name_parts = os.path.splitext(filename)[0].split(' - ')
        if len(name_parts) >= 2:
            author = name_parts[0].strip()
            title = name_parts[1].strip()
        else:
            title = name_parts[0].strip()
            author = "Unknown Author"
            
        out_filename = f"{title} - {author}.txt".replace('  ', ' ')
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove scripts/styles
        for s in soup(["script", "style"]):
            s.extract()
            
        # Get raw text
        raw_text = soup.get_text()
        
        # Clean and Wrap
        final_text = clean_text_wrapper(raw_text)
        
        # Build Output
        full_content = []
        full_content.append(f"*** START OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***")
        full_content.append(f"\nTitle: {title}")
        full_content.append(f"Author: {author}")
        full_content.append("\n" + "-"*40 + "\n")
        full_content.append(final_text)
        full_content.append(f"\n*** END OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(full_content))
            
        # Clean up temp files
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        logging.error(f"Failed to convert {mobi_path}: {e}")
        return False

def main():
    mobi_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith('.mobi'):
                mobi_files.append(os.path.join(root, file))
    
    print(f"Found {len(mobi_files)} MOBI files.")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(convert_mobi, mobi_files))
        
    print(f"Converted {sum(results)}/{len(mobi_files)} books.")

if __name__ == "__main__":
    main()
