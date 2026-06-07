# ================================================================
# smart_serializer.py (v3.3 — two-phase init, supports unknown title_slug)
#
# Handles:
#   - Creating temporary folder before title exists
#   - Renaming folder once title_slug is known
#   - Saving metadata, prompts, full book text
#   - ED64 memory dumps
# ================================================================

import os
import json
import re
from datetime import datetime


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def sanitize_filename(name):
    name = name.replace("\n", " ").replace("\r", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)

    # protect filesystem
    return name[:96].lower()   # 96 chars max



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------
# Serializer (v3.3)
# ---------------------------------------------------------------
class SmartSerializer:
    """
    Filesystem controller for longbook generation.
    """

    def __init__(self, root_dir):
        """
        self.base_dir = Path("generated_texts_og")/
        """
        self.root_dir = root_dir

        self.texts_dir = os.path.join(root_dir, "Texts")
        self.prompts_dir = os.path.join(root_dir, "prompts")

        ensure_dir(self.texts_dir)
        ensure_dir(self.prompts_dir)

        self.debug_root = None        # actual folder used for debug dumps
        self._pending_timestamp = None
        self._pending_temp_dir = None

    # -----------------------------------------------------------
    # PHASE 1 — temporary folder, BEFORE title is known
    # -----------------------------------------------------------
    def init_temp_book_folder(self, timestamp, custom_folder_name=None):
        """
        Creates prompts/{folder_name}/ as a temporary debug folder.
        If custom_folder_name is provided (e.g. 'run_11'), uses that.
        Otherwise uses timestamp.
        """
        self._pending_timestamp = timestamp
        
        folder_name = custom_folder_name if custom_folder_name else timestamp
        folder_path = os.path.join(self.prompts_dir, folder_name)

        ensure_dir(folder_path)
        self._pending_temp_dir = folder_path
        self.debug_root = folder_path

        return folder_path

    # -----------------------------------------------------------
    # PHASE 2 — rename after title_slug is known
    # -----------------------------------------------------------
    def finalize_book_folder(self, title_slug, genre=None, conditioning_mode=None):
        """
        Renames prompts/{timestamp}/ -> prompts/{timestamp}_{title_slug}_{genre}_{conditioning_mode}/
        and sets that as the active debug_root.
        """
        if self._pending_timestamp is None:
            raise RuntimeError("init_temp_book_folder() must be called first.")

        # Build final folder name
        genre_slug = sanitize_filename(genre) if genre else ""
        mode_slug = sanitize_filename(conditioning_mode) if conditioning_mode else ""

        name_parts = [self._pending_timestamp, sanitize_filename(title_slug)]
        if genre_slug:
            name_parts.append(genre_slug)
        if mode_slug:
            name_parts.append(mode_slug)
        final_name = "_".join(name_parts)
        
        final_path = os.path.join(self.prompts_dir, final_name)

        # rename only if different
        if self._pending_temp_dir != final_path:
            os.rename(self._pending_temp_dir, final_path)

        self.debug_root = final_path
        return final_path

    # -----------------------------------------------------------
    # METADATA JSON
    # -----------------------------------------------------------
    def save_metadata(self, metadata_dict):
        if self.debug_root is None:
            raise RuntimeError("Book folder not initialized.")

        path = os.path.join(self.debug_root, "metadata.json")
        with open(path, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        return path

    # -----------------------------------------------------------
    # PROMPT DEBUGGING
    # -----------------------------------------------------------
    def save_debug_prompt(self, name, content):
        if self.debug_root is None:
            raise RuntimeError("Book folder not initialized.")

        sanitized = sanitize_filename(name)
        path = os.path.join(self.debug_root, f"{sanitized}.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return path

    # -----------------------------------------------------------
    # BOOK TEXT
    # -----------------------------------------------------------
    def save_book_text(self, model_name, title_slug, broader_genre, timestamp, text):
        model_slug = sanitize_filename(model_name.replace("/", "_"))
        genre_slug = sanitize_filename(broader_genre)
        title_slug = sanitize_filename(title_slug)

        filename = f"LONGBOOK__{model_slug}__{genre_slug}__{title_slug}__{timestamp}.txt"
        path = os.path.join(self.texts_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        return path

    # -----------------------------------------------------------
    # MEMORY SNAPSHOT
    # -----------------------------------------------------------
    def save_memory_snapshot(self, memory_manager, timestamp, full_tokens=None, accumulated_metrics=None, graph_memory=None):
        if self.debug_root is None:
            raise RuntimeError("Book folder not initialized.")

        folder = os.path.join(self.debug_root, f"memory_{timestamp}")
        ensure_dir(folder)

        mem = memory_manager.to_serializable()
        if graph_memory is not None:
            mem["graph_memory"] = graph_memory
        path = os.path.join(folder, "memory.json")

        with open(path, "w") as f:
            json.dump(mem, f, indent=2)

        # Updated: Persist full tokens as binary for instant resume
        if full_tokens is not None:
            import numpy as np
            token_path = os.path.join(folder, "tokens.npy")
            np.save(token_path, np.array(full_tokens, dtype=np.int32))

        # Updated: Persist accumulated metrics to avoid re-calculation
        if accumulated_metrics is not None:
            metric_path = os.path.join(folder, "accumulated_metrics.json")
            # Convert numpy/tensor values to float for JSON
            clean_metrics = {k: float(v) for k, v in accumulated_metrics.items()}
            with open(metric_path, "w") as f:
                json.dump(clean_metrics, f, indent=2)

        return path

    # -----------------------------------------------------------
    # MEMORY JSON BUILDER (M-inject3)
    # -----------------------------------------------------------
    def build_memory_json_string(self, summary_manager, graph_memory, local_context_text):
        """
        Build the structured JSON memory block injected into each generation prompt.
        This does NOT write anything to disk – it only returns the JSON string.
        """

        # Chunk summaries (S2)
        chunk_summaries = summary_manager.chunks

        # Episodic summaries (E2)
        episodic_summaries = summary_manager.episodes

        # Global summary (G3)
        global_summary = summary_manager.global_summary

        # Graph memory
        graph_state = graph_memory.to_serializable()

        # Local textual context
        local_context = local_context_text[-2000:]  # trim long context for safety

        mem = {
            "memory": {
                "S2_chunk_summaries": chunk_summaries,
                "E2_episodic_summaries": episodic_summaries,
                "G3_global_summary": global_summary,
                "graph_memory": graph_state,
                "local_context": local_context,
            }
        }

        return json.dumps(mem, indent=2)
