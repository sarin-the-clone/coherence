import sys
import os
import requests
import tiktoken

def get_gutenberg_text(book_id):
    """Download text from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.text) > 1000:
            raw_text = r.text
            
            # Trim boilerplate
            start = raw_text.find("*** START")
            end = raw_text.find("*** END")
            if start != -1 and end != -1:
                body = raw_text[start:end].strip()
            else:
                body = raw_text.strip()
                
            return body
    except requests.RequestException as e:
        print(f"Error fetching {book_id}: {e}")
        pass
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_tokens.py <book_id>")
        sys.exit(1)
        
    book_id = sys.argv[1]
    print(f"Fetching book {book_id}...")
    
    text = get_gutenberg_text(book_id)
    if not text:
        print(f"Failed to fetch book {book_id}")
        sys.exit(1)
        
    print(f"Downloaded text of length {len(text)} characters.")
    
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    
    print(f"Book {book_id} has {len(tokens)} tokens.")

if __name__ == "__main__":
    main()
