import json
import glob
from pathlib import Path

prompts_dir = Path("generated_texts_og/prompts")
out_dir = Path("generated_texts_og/Texts")
out_dir.mkdir(parents=True, exist_ok=True)

folders = [f for f in prompts_dir.glob("*") if f.is_dir() and not f.name.startswith("_")]

count = 0
for folder in folders:
    title = folder.name
    out_file = out_dir / f"{title}.txt"
        
    chunks = sorted(folder.glob("chunk_*_attempt_*_accepted.txt"))
    if not chunks:
        all_chunks = sorted(folder.glob("chunk_*_attempt_*.txt"))
        if not all_chunks:
            continue
            
        chunk_dict = {}
        for c in all_chunks:
            parts = c.stem.split('_')
            try:
                c_idx = int(parts[1])
                a_idx = int(parts[3])
                if c_idx not in chunk_dict or a_idx > chunk_dict[c_idx][0]:
                    chunk_dict[c_idx] = (a_idx, c)
            except:
                pass
        chunks = [v[1] for k, v in sorted(chunk_dict.items())]
        
    full_text = []
    for c in chunks:
        try:
            with open(c, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        # API response format usually has 'text' or 'response' inside 'choices'
                        if "choices" in data and len(data["choices"]) > 0:
                            msg = data["choices"][0].get("message", {})
                            text = msg.get("content", "")
                            if text: full_text.append(text)
                        elif "text" in data:
                            full_text.append(data["text"])
                        elif "content" in data:
                            full_text.append(data["content"])
                        elif "response" in data:
                            full_text.append(data["response"])
                        else:
                            # Dump the json as string if we can't find the text field
                            full_text.append(json.dumps(data, indent=2))
                    else:
                        full_text.append(content)
                except json.JSONDecodeError:
                    # Raw text
                    full_text.append(content)
        except Exception as e:
            print(f"Error reading {c}: {e}")
            
    if full_text:
        with open(out_file, 'w', encoding='utf-8') as out:
            out.write("\n\n".join(full_text))
        count += 1
        print(f"Stitched {title}.txt ({len(chunks)} chunks)")

print(f"Stitched {count} new books.")
