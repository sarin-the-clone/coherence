import argparse
import os
import sys

# Make sure project root is available
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the EPUB exporter from run_longbook.py
from generated_texts_og.run_longbook import save_book_to_epub


def auto_extract_title(filename: str) -> str:
    """
    Extracts title_slug from LONGBOOK__model__genre__title_slug__timestamp.txt
    """
    base = os.path.basename(filename)
    parts = base.split("__")

    if len(parts) >= 4:
        return parts[3].replace("_", " ").title()

    # fallback
    return base.replace(".txt", "").replace("_", " ").title()


def main():
    parser = argparse.ArgumentParser(description="Export an existing longbook TXT to EPUB.")

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input .txt book.")

    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional output .epub path.")

    parser.add_argument("--title", "-t", type=str, default=None,
                        help="Override title. If omitted, auto-extracted from filename.")

    parser.add_argument("--author", "-a", type=str, default="AI_LLM",
                        help="Author for EPUB metadata.")

    args = parser.parse_args()

    txt_path = args.input
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"TXT not found: {txt_path}")

    # Determine title
    title = args.title if args.title else auto_extract_title(txt_path)

    # Determine output path
    if args.output:
        epub_path = args.output
    else:
        epub_dir = os.path.join(PROJECT_ROOT, "Epub")
        os.makedirs(epub_dir, exist_ok=True)

        slug = title.replace(" ", "_").lower()
        epub_path = os.path.join(epub_dir, f"{slug}.epub")

    print(f"Exporting to: {epub_path}")

    # Call your main EPUB generator
    save_book_to_epub(
        txt_path=txt_path,
        epub_path=epub_path,
        title=title,
        author=args.author
    )


if __name__ == "__main__":
    main()
