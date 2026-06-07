"""
run_longbook.py

Main entry point for generating long-form books using LLMs.
Handles CLI arguments, model initialization, generation loop activation,
and post-processing (EPUB creation).
"""
import argparse
import os
import sys
import yaml
import time

# Mitigate CUDA OOM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure project root is on path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# If running from generated_texts/, parent is root. If running from root, THIS_DIR is root/generated_texts
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from generated_texts_og.longbook_engine import LongBookGenerator

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.platypus import Spacer
from reportlab.lib.units import cm

from reportlab.lib.enums import TA_JUSTIFY

from ebooklib import epub


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_book_to_epub(
    txt_path: str,
    epub_path: str,
    title: str = "Generated Book",
    author: str = "AI_LLM",
):
    """
    Convert a generated text file into a reflowable EPUB.
    """

    # Read text
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create EPUB book
    book = epub.EpubBook()

    # Metadata
    book.set_title(title)
    book.add_author(author)

    # Split into paragraphs (EPUB wants HTML blocks)
    paragraphs = text.split("\n\n")

    # Create chapters (simple: one big chapter, but reflowable)
    chapter_html = "<h1>{}</h1>".format(title)
    for p in paragraphs:
        p = p.strip().replace("\n", "<br/>")
        if p:
            chapter_html += f"<p>{p}</p>\n"

    chapter = epub.EpubHtml(title=title, file_name="chap_1.xhtml", lang="en")
    chapter.set_content(chapter_html)

    book.add_item(chapter)

    # Table of contents
    book.toc = (epub.Link('chap_1.xhtml', title, 'chap_1'),)

    # Add the default NCX and Nav files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # CSS for nicer reading
    style = '''
    body {
        font-family: serif;
        font-size: 1.0em;
        line-height: 1.6;
        margin: 0 5% 0 5%;
    }
    p {
        text-indent: 1.2em;
        margin-top: 0;
        margin-bottom: 1em;
    }
    '''
    nav_css = epub.EpubItem(
        uid="style_nav",
        file_name="style/nav.css",
        media_type="text/css",
        content=style.encode("utf-8")
    )
    book.add_item(nav_css)

    # Spine
    book.spine = ['nav', chapter]

    # Write to file
    epub.write_epub(epub_path, book)
    print(f"EPUB saved → {epub_path}")



# def save_book_to_pdf(
#     txt_path: str,
#     pdf_path: str,
#     title: str = "Generated Book",
#     font_size: int = 11,
# ):
#     """
#     Convert a generated text file into a PDF.
#     """
#
#     # Read the book text
#     with open(txt_path, "r", encoding="utf-8") as f:
#         text = f.read()
#
#     # Setup document
#     doc = SimpleDocTemplate(
#         pdf_path,
#         pagesize=A4,
#         leftMargin=2 * cm,
#         rightMargin=2 * cm,
#         topMargin=2 * cm,
#         bottomMargin=2 * cm,
#     )
#
#     styles = getSampleStyleSheet()
#     story = []
#
#     # Title
#     style_title = styles["Title"]
#     style_title.fontSize = font_size + 6
#     story.append(Paragraph(title, style_title))
#     story.append(Spacer(1, 1 * cm))
#
#     # Body text
#     style_body = styles["BodyText"]
#     style_body.fontSize = font_size
#     style_body.leading = font_size * 1.3
#     style_body.alignment = TA_JUSTIFY
#
#     # Split into paragraphs to avoid giant text block issues
#     for paragraph in text.split("\n\n"):
#         clean_p = paragraph.strip().replace("\n", "<br/>")
#         if not clean_p:
#             continue
#         story.append(Paragraph(clean_p, style_body))
#         story.append(Spacer(1, 0.5 * cm))
#
#     # Build PDF
#     try:
#         doc.build(story)
#         print(f"PDF saved → {pdf_path}")
#     except Exception as e:
#         print(f"Failed to save PDF: {e}")

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None, # Changed from required=True to allow prompt_file
        help="Initial high-level prompt that defines the book theme."
    )
    
    parser.add_argument(
        "--prompt_file", "-f",
        type=str,
        default=None,
        help="Path to markdown file containing the initial prompt."
    )

    parser.add_argument(
        "--chunks", "-n",
        type=int,
        default=12,
        help="How many long chunks to generate (each chunk ~1–2k tokens)."
    )

    parser.add_argument(
        "--target_tokens",
        type=int,
        default=None,
        help="Target total book size in tokens (overrides chunks)."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help="Model key from config (qwen, llama, mistral)."
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(THIS_DIR, "config", "model_config.yaml"),
        help="Path to model config file."
    )

    parser.add_argument(
        "--max_tokens", type=int, default=20000,
        help="Max new tokens per chunk."
    )

    parser.add_argument(
        "--temp_low", type=float, default=None,
        help="Temperature minimum for sweeps."
    )
    parser.add_argument(
        "--temp_high", type=float, default=None,
        help="Temperature maximum for sweeps."
    )

    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling probability (default: 0.95 for higher entropy)."
    )

    parser.add_argument(
        "--entropy_min", type=float, default=None,
        help="Override minimum entropy threshold (gating)."
    )

    parser.add_argument(
        "--disable_gating", action="store_true",
        help="Disable metric-based gating (sanity checks remain)."
    )

    parser.add_argument(
        "--metric_gating_patience", type=int, default=20,
        help=(
            "Suspend metric-based gating after this many consecutive chunks "
            "require metric-gated retries. Sanity checks remain active. "
            "Use 0 to disable this circuit breaker."
        )
    )

    parser.add_argument(
        "--conditioning_mode",
        choices=[
            "basic_conditioning",
            "summaries_conditioning",
            "extended_conditioning",
            "minimal_fixed_conditioning",
            "markov_rolling_conditioning",
            "minimal_rolling_conditioning",
        ],
        default="extended_conditioning",
        help=(
            "Prompt conditioning mode: "
            "basic_conditioning (C + I + D_n + L_n), "
            "summaries_conditioning (basic + M_n), "
            "extended_conditioning (summaries + G_n + R_n), "
            "minimal_fixed_conditioning (C + I), or "
            "markov_rolling_conditioning (first chunk: C + I, then only L_n). "
            "minimal_rolling_conditioning is accepted as a legacy alias."
        ),
    )

    parser.add_argument(
        "--ppl_min", type=float, default=None,
        help="Override minimum perplexity threshold (gating)."
    )

    parser.add_argument(
        "--restart_every", type=int, default=None,
        help="Restart process after N chunks to clear VRAM fragmentation (exit code 42)."
    )
    
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1,
        help="Repetition penalty for generation (default: 1.1). Try 1.05-1.2 for different models."
    )
    
    parser.add_argument(
        "--n_attempts", type=int, default=3,
        help="Number of regeneration attempts per chunk before moving on (default: 3)."
    )

    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing run directory to resume from."
    )

    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Fixed persistent identifier for this run (e.g., 'run_11'). Folder will be named this until completion."
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.model not in config["models"]:
        print(f"Error: Model '{args.model}' not found in config.")
        print(f"Available models: {list(config['models'].keys())}")
        return

    model_config = config["models"][args.model]

    # Allow args to override config for max_tokens
    max_tokens = args.max_tokens if args.max_tokens != 20000 else model_config.get("max_tokens", 2048)

    # Determine temperature range
    # 1. User args (if explicitly provided, unlikely to detect default unless we change argparse default to None)
    # 2. Config "temperature_range"
    # 3. Config "temperature" (legacy)
    # 4. Fallback (0.8, 1.3)
    
    # We change argparse defaults to None to detect user input
    
    if args.temp_low is not None and args.temp_high is not None:
        temp_range = (args.temp_low, args.temp_high)
    else:
        # Check config
        if "temperature_range" in model_config:
            temp_range = tuple(model_config["temperature_range"])
        elif "temperature" in model_config:
            t = model_config["temperature"]
            temp_range = (t, t + 0.3) # Approximate range
        else:
            temp_range = (0.8, 1.3) # Fallback

    generator = LongBookGenerator(
        model_name=args.model,
        max_chunk_tokens=max_tokens,
        temperature_range=temp_range,

        top_p=args.top_p,
        entropy_threshold=args.entropy_min,
        ppl_threshold=args.ppl_min,
        disable_gating=args.disable_gating,
        metric_gating_patience=args.metric_gating_patience,
        conditioning_mode=args.conditioning_mode,
        restart_every=args.restart_every,
        
        repetition_penalty=args.repetition_penalty,
        n_attempts=args.n_attempts,
    )

    # Handle Prompt Source
    prompt_text = args.prompt
    prompt_filename = "Manual Entry"

    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            prompt_filename = os.path.basename(args.prompt_file)
        else:
            print(f"Error: Prompt file not found: {args.prompt_file}")
            return
            
    if not prompt_text:
        print("Error: Must provide either --prompt or --prompt_file")
        return

    out_path = generator.generate_book(
        initial_prompt=prompt_text,
        n_chunks=args.chunks,
        target_tokens=args.target_tokens,
        resume_from_dir=args.resume,
        run_id=args.run_id,
        prompt_filename=prompt_filename  # Pass filename
    )

    print("\n==========================")
    print("📚 Book saved at:")
    print(out_path)
    print("==========================")
    print("Done.")
    elapsed = time.time() - start_time
    print(f"⏱ Runtime for LongBook: {elapsed/3600:.2f} hours or {elapsed/60:.2f} minutes")
    
    try:
        # Extract title slug from filename (format: LONGBOOK__model__genre__title_slug__timestamp.txt)
        # We split by "__" and take the 4th element (index 3)
        parts = os.path.basename(out_path).split("__")
        if len(parts) >= 4:
            title_slug = parts[3]
        else:
            # Fallback if naming convention changes
            title_slug = "generated_book_" + parts[-1].replace(".txt", "")
            
        pdf_dir = os.path.join(THIS_DIR, "Pdf")
        os.makedirs(pdf_dir, exist_ok=True)
        epub_dir = os.path.join(THIS_DIR, "Epub")
        os.makedirs(epub_dir, exist_ok=True)
        
        # pdf_path = os.path.join(pdf_dir, f"{title_slug}.pdf")
        epub_path = os.path.join(epub_dir, f"{title_slug}.epub")

        # save_book_to_pdf(out_path, pdf_path, title_slug.replace("_", " ").title())
        save_book_to_epub(out_path, epub_path, title_slug.replace("_", " ").title(), author=args.model)

    except Exception as e:
        print(f"Could not generate PDF or epub: {e}")
    
if __name__ == "__main__":
    main()
