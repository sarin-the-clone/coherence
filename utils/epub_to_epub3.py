#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import zipfile
import shutil
import uuid
import tempfile
import argparse
import subprocess
import re
import time
import html
import posixpath
import math
import asyncio
import tempfile
from typing import Optional, Dict

from bs4 import BeautifulSoup
from ebooklib import epub
from lxml import etree
try:
    import edge_tts
except ImportError:
    edge_tts = None
try:
    from gtts import gTTS
except ImportError:
    gTTS = None


# ============================================================
# Helpers
# ============================================================

def unzip_epub(epub_path, extract_to):
    with zipfile.ZipFile(epub_path, "r") as z:
        z.extractall(extract_to)


def extract_text_from_xhtml(path):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    return soup.get_text("\n")


def chunk_text(text: str, limit: int = 2000):
    """
    Split text into chunks <= limit chars, preferring sentence boundaries.
    """
    sentences = re.split(r"(?<=[\\.\\!\\?])\\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if not s:
            continue
        if len(current) + len(s) + 1 <= limit:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            if len(s) <= limit:
                current = s
            else:
                # hard split long sentence
                for i in range(0, len(s), limit):
                    part = s[i:i + limit]
                    chunks.append(part)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def tts_espeak(text, out_mp3, voice="en-gb+m3"):
    print(f"→ eSpeak-NG ({voice}) generating: {out_mp3}")

    CHUNK = 500
    parts = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]

    tmp_wav = out_mp3.replace(".mp3", ".wav")
    os.makedirs(os.path.dirname(tmp_wav), exist_ok=True)

    with open(tmp_wav, "wb") as wav:
        for p in parts:
            subprocess.run(
                ["espeak-ng", "-v", voice, "--stdout", p],
                stdout=wav,
                stderr=subprocess.DEVNULL,
            )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            tmp_wav,
            "-codec:a",
            "libmp3lame",
            "-ar",
            "44100",
            "-ac",
            "1",
            "-b:a",
            "128k",
            out_mp3,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    os.remove(tmp_wav)
    return out_mp3


async def _edge_tts_stream(text: str, out_mp3: str, voice: str):
    communicator = edge_tts.Communicate(text, voice)
    os.makedirs(os.path.dirname(out_mp3), exist_ok=True)
    with open(out_mp3, "wb") as f:
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])


def tts_edge(text, out_mp3, voice="en-US-AriaNeural", chunk_limit=800, _retry=False):
    if edge_tts is None:
        raise RuntimeError("edge-tts not installed. Install with `pip install edge-tts`.")
    print(f"→ edge-tts ({voice}) generating: {out_mp3}")
    tmp_dir = tempfile.mkdtemp()
    parts = chunk_text(text, limit=chunk_limit)
    mp3_parts = []
    for idx, chunk in enumerate(parts, start=1):
        part_path = os.path.join(tmp_dir, f"part_{idx}.mp3")
        try:
            asyncio.run(_edge_tts_stream(chunk, part_path, voice))
            mp3_parts.append(part_path)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not _retry and chunk_limit > 400:
                # retry once with smaller chunks
                print("   edge-tts failed; retrying with smaller chunks...")
                return tts_edge(text, out_mp3, voice, chunk_limit=400, _retry=True)
            raise RuntimeError(f"edge-tts failed on chunk {idx}: {e}")

    if len(mp3_parts) == 1:
        shutil.move(mp3_parts[0], out_mp3)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return out_mp3

    # Concatenate using ffmpeg concat demuxer
    list_file = os.path.join(tmp_dir, "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in mp3_parts:
            f.write(f"file '{p}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_mp3],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_mp3


def tts_gtts(text, out_mp3, lang="en"):
    if gTTS is None:
        raise RuntimeError("gTTS not installed. Install with `pip install gTTS`.")
    print(f"→ gTTS ({lang}) generating: {out_mp3}")
    os.makedirs(os.path.dirname(out_mp3), exist_ok=True)
    gTTS(text=text, lang=lang).save(out_mp3)
    return out_mp3


def shift_pitch(audio_path: str, semitones: float):
    """
    Adjust pitch by semitones (can be negative). Uses ffmpeg.
    """
    if not semitones:
        return
    factor = 2 ** (semitones / 12.0)
    tmp_out = audio_path + ".pitch.tmp.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-af",
        f"asetrate=44100*{factor},aresample=44100,atempo={1/factor}",
        "-codec:a",
        "libmp3lame",
        audio_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ffmpeg overwrote in-place; nothing else to do
    return audio_path


def create_smil(xhtml_file, audio_file, smil_output, segments=None):
    NS = {
        None: "http://www.w3.org/ns/SMIL",
        "epub": "http://www.idpf.org/2007/ops"
    }

    smil = etree.Element("smil", nsmap=NS)
    body = etree.SubElement(smil, "body")
    seq = etree.SubElement(
        body, "seq",
        attrib={"{http://www.idpf.org/2007/ops}textref": posixpath.relpath(xhtml_file, "smil")}
    )

    # Paths in SMIL are relative to the SMIL directory
    text_href_base = posixpath.relpath(xhtml_file, "smil")
    audio_href = posixpath.relpath(audio_file, "smil")

    if not segments:
        par = etree.SubElement(seq, "par")
        etree.SubElement(par, "text", attrib={"src": text_href_base})
        etree.SubElement(par, "audio", attrib={"src": audio_href})
    else:
        for frag_id, clip_begin, clip_end in segments:
            par = etree.SubElement(seq, "par")
            etree.SubElement(par, "text", attrib={"src": f"{text_href_base}#{frag_id}"})
            audio_attribs = {"src": audio_href}
            if clip_begin is not None:
                audio_attribs["clipBegin"] = clip_begin
            if clip_end is not None:
                audio_attribs["clipEnd"] = clip_end
            etree.SubElement(par, "audio", attrib=audio_attribs)

    with open(smil_output, "wb") as f:
        f.write(etree.tostring(
            smil,
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8",
        ))


# ============================================================
# Metadata helpers
# ============================================================

def extract_metadata(work_root: str) -> Dict[str, Optional[str]]:
    """
    Pull basic metadata from the first OPF we find.
    """
    meta = {"title": None, "creator": None, "language": None, "identifier": None}
    ns = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "opf": "http://www.idpf.org/2007/opf",
    }

    for root, _, files in os.walk(work_root):
        for f in files:
            if not f.lower().endswith(".opf"):
                continue
            opf_path = os.path.join(root, f)
            try:
                tree = etree.parse(opf_path)
                meta["title"] = (tree.findtext(".//dc:title", namespaces=ns) or "").strip() or None
                meta["creator"] = (tree.findtext(".//dc:creator", namespaces=ns) or "").strip() or None
                meta["language"] = (tree.findtext(".//dc:language", namespaces=ns) or "").strip() or None
                meta["identifier"] = (tree.findtext(".//dc:identifier", namespaces=ns) or "").strip() or None
                return meta
            except Exception:
                continue
    return meta


def ffprobe_duration(path: str) -> Optional[float]:
    """
    Return duration in seconds using ffprobe. None on failure.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def fmt_duration(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.mmm for media:duration.
    """
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ============================================================
# Chapter reconstruction — 40 paragraphs per chapter
# ============================================================

def split_into_chapters(full_text, paragraphs_per_chapter=40):
    cleaned = re.sub(r"\n\s*\n+", "\n", full_text).strip()
    paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]

    chapters = []
    for i in range(0, len(paragraphs), paragraphs_per_chapter):
        part = paragraphs[i:i + paragraphs_per_chapter]
        chapters.append("\n\n".join(part))
    return chapters


# ============================================================
# MAIN
# ============================================================

def convert_epub_to_epub3(
    epub_path,
    output_path=None,
    voice="en-gb+m3",
    tts_backend="espeak",
    edge_chunk_limit=800,
    edge_fallback_espeak=True,
    edge_fallback_voice="en-gb+m3",
    gtts_lang="en",
    pitch_shift=0.0,
):
    start_time = time.time()

    # Output file
    if output_path is None:
        output_path = epub_path.replace(".epub", "_with_audio.epub")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("→ Unzipped EPUB to:", end=" ")
    work = tempfile.mkdtemp()
    unzip_epub(epub_path, work)
    print(work)

    # Normalize structure → ensure a single OEBPS root
    oebps = os.path.join(work, "OEBPS")
    if not os.path.exists(oebps):
        os.makedirs(oebps)
        for item in os.listdir(work):
            if item in ("META-INF", "mimetype", "OEBPS"):
                continue
            shutil.move(os.path.join(work, item), oebps)

    # ----------------------------------------------------------
    # 1. Locate main XHTML content file (chap_1.xhtml)
    # ----------------------------------------------------------
    chap1_path = None
    for root, _, files in os.walk(work):
        for f in files:
            if f.lower().endswith(".xhtml") and "chap_1" in f.lower():
                chap1_path = os.path.join(root, f)

    if not chap1_path:
        raise RuntimeError("chap_1.xhtml not found in EPUB — cannot extract text.")

    print("→ Using main source file:", chap1_path)

    # Base metadata from original EPUB if present
    meta = extract_metadata(work)

    # Extract text
    full_text = extract_text_from_xhtml(chap1_path)
    print(f"→ Extracted {len(full_text)} characters.")

    # ----------------------------------------------------------
    # 2. Split into chapters (40 paragraphs each)
    # ----------------------------------------------------------
    chapters = split_into_chapters(full_text, paragraphs_per_chapter=40)
    chapters = [c for c in chapters if c.strip()]
    print(f"→ Constructed {len(chapters)} chapters.")

    # ----------------------------------------------------------
    # 3. Build new EPUB structure
    # ----------------------------------------------------------
    book = epub.EpubBook()
    book.set_identifier(meta.get("identifier") or str(uuid.uuid4()))
    book.set_language(meta.get("language") or "en")
    book.set_title(meta.get("title") or "EPUB with Audio Overlays")
    book.add_author(meta.get("creator") or "Unknown")
    # Helps some readers highlight during playback
    book.add_metadata(
        "http://www.idpf.org/epub/vocab/overlays/#",
        "active-class",
        "-epub-media-overlay-active",
    )
    book.add_metadata(
        "http://www.idpf.org/epub/vocab/overlays/#",
        "narrator",
        voice,
    )

    spine = ["nav"]
    toc_items = []
    smil_durations = {}

    audio_dir = os.path.join(oebps, "audio")
    smil_dir = os.path.join(oebps, "smil")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(smil_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 4. Create chapters + audio + smil
    # ----------------------------------------------------------
    for chap_id, chap_text in enumerate(chapters, start=1):
        clean = chap_text.strip()
        print(f"→ Chapter {chap_id}: {len(clean)} chars")

        if not clean:
            continue

        # HTML paragraph split with stable ids so SMIL can target fragments
        escaped = html.escape(clean)
        pieces = [p for p in escaped.split("\n") if p.strip()]
        para_ids = [f"p{chap_id}_{i}" for i in range(1, len(pieces) + 1)]
        paras = "".join(
            f'<p id="{pid}">{text}</p>'
            for pid, text in zip(para_ids, pieces)
        )

        # XHTML content
        filename = f"chapter_{chap_id}.xhtml"
        xhtml_path = os.path.join(oebps, filename)

        # Note: ebooklib's lxml HTML parser chokes on XML declarations in Unicode
        # strings, so we omit the leading <?xml ...?> header here.
        xhtml = f"""<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter {chap_id}</title></head>
<body>
<h1>Chapter {chap_id}</h1>
<p>{paras}</p>
</body>
</html>
"""

        with open(xhtml_path, "w", encoding="utf-8") as f:
            f.write(xhtml)

        # Add XHTML item
        chap_item = epub.EpubHtml(
            uid=f"chap_{chap_id}",
            file_name=filename,
            content=xhtml,
        )
        chap_item.properties.append("media-overlay")

        book.add_item(chap_item)
        spine.append(chap_item)
        toc_items.append(chap_item)

        # AUDIO
        mp3_path = os.path.join(audio_dir, f"chapter_{chap_id}.mp3")
        if tts_backend == "edge":
            try:
                tts_edge(clean, mp3_path, voice, chunk_limit=edge_chunk_limit)
            except Exception as e:
                if not edge_fallback_espeak:
                    raise
                print(f"   edge-tts failed ({e}); falling back to espeak-ng.")
                tts_espeak(clean, mp3_path, edge_fallback_voice)
        elif tts_backend == "gtts":
            tts_gtts(clean, mp3_path, lang=gtts_lang)
        else:
            tts_espeak(clean, mp3_path, voice)
        if pitch_shift:
            shift_pitch(mp3_path, pitch_shift)

        audio_item = epub.EpubItem(
            uid=f"audio_{chap_id}",
            file_name=f"audio/chapter_{chap_id}.mp3",
            media_type="audio/mpeg",
            content=open(mp3_path, "rb").read(),
        )
        book.add_item(audio_item)

        # SMIL with rough per-paragraph timings so readers can sync/highlight
        smil_file = os.path.join(smil_dir, f"chapter_{chap_id}.smil")
        dur_seconds = ffprobe_duration(mp3_path)
        segments = []
        if dur_seconds and para_ids:
            total_chars = sum(len(p) for p in pieces) or 1
            elapsed = 0.0
            for pid, text in zip(para_ids, pieces):
                frac = len(text) / total_chars
                seg_dur = dur_seconds * frac
                clip_begin = fmt_duration(elapsed)
                elapsed += seg_dur
                clip_end = fmt_duration(min(elapsed, dur_seconds))
                segments.append((pid, clip_begin, clip_end))
            # ensure final segment ends exactly at duration
            if segments:
                segments[-1] = (segments[-1][0], segments[-1][1], fmt_duration(dur_seconds))
        create_smil(filename, f"audio/chapter_{chap_id}.mp3", smil_file, segments=segments)

        smil_uid = f"smil_{chap_id}"
        smil_item = epub.EpubItem(
            uid=smil_uid,
            file_name=f"smil/chapter_{chap_id}.smil",
            media_type="application/smil+xml",
            content=open(smil_file, "rb").read(),
        )
        book.add_item(smil_item)

        # Link overlay to SMIL item id (per EPUB3 spec)
        chap_item.media_overlay = smil_uid

        # Record duration for metadata refinement
        if dur_seconds:
            smil_durations[smil_uid] = dur_seconds

    # ----------------------------------------------------------
    # 5. Add navigation + save EPUB
    # ----------------------------------------------------------
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    if toc_items:
        book.toc = tuple(toc_items)

    # media:duration metadata refinements (helps some readers)
    total_seconds = 0.0
    for smil_uid, seconds in smil_durations.items():
        total_seconds += seconds
        book.add_metadata(
            "http://idpf.org/epub/vocab/media/#",
            "duration",
            fmt_duration(seconds),
            {"refines": f"#{smil_uid}"},
        )
    if total_seconds > 0:
        book.add_metadata(
            "http://idpf.org/epub/vocab/media/#",
            "duration",
            fmt_duration(total_seconds),
        )

    epub.write_epub(output_path, book, {})

    shutil.rmtree(work)

    elapsed = time.time() - start_time
    print(f"\n✔ EPUB3 with audio overlays saved to: {output_path}")
    print(f"⏱ Runtime: {elapsed/60:.2f} minutes")
    print("Done.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--voice", default="en-gb+m3", help="Voice identifier for TTS backend.")
    parser.add_argument("--tts", choices=["espeak", "edge", "gtts"], default="gtts", help="TTS backend to use.")
    parser.add_argument("--edge-chunk-limit", type=int, default=800, help="Approximate max chars per edge-tts request.")
    parser.add_argument("--edge-no-fallback", action="store_true", help="Disable falling back to espeak if edge-tts fails.")
    parser.add_argument("--edge-fallback-voice", default="en-gb+m3", help="Voice to use if edge-tts fails and fallback is enabled.")
    parser.add_argument("--gtts-lang", default="en", help="Language code for gTTS (e.g., en, en-us, en-uk).")
    parser.add_argument("--pitch-shift", type=float, default=3.0, help="Pitch shift in semitones (negative lowers pitch).")
    args = parser.parse_args()

    convert_epub_to_epub3(
        epub_path=args.input,
        output_path=args.output,
        voice=args.voice,
        tts_backend=args.tts,
        edge_chunk_limit=args.edge_chunk_limit,
        edge_fallback_espeak=not args.edge_no_fallback,
        edge_fallback_voice=args.edge_fallback_voice,
        gtts_lang=args.gtts_lang,
        pitch_shift=args.pitch_shift,
    )


if __name__ == "__main__":
    main()
