# ================================================================
# longbook_engine.py (v4.0 - Model Agnostic)
# ================================================================

import os
import json
import yaml
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import sys
import unicodedata
import re
import gc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Memories
from generated_texts_og.memory.local_context import LocalContext
from generated_texts_og.memory.summary_manager import SummaryManager
from generated_texts_og.memory.graph_memory import GraphMemory
from generated_texts_og.memory.semantic_memory import SemanticMemory
from generated_texts_og.memory.compression_transformer import CompressionTransformer
from generated_texts_og.memory.smart_serializer import SmartSerializer

# Utils
from generated_texts_og.utils.chunk_tools import extract_new_text
from generated_texts_og.utils.coherence_decision import is_chunk_acceptable
from generated_texts_og.models.model_interface import ModelFactory
from generated_texts_og.memory.memory_summarizer import MemorySummarizer
from generated_texts_og.utils.title_and_genre import generate_title_and_genre
from generated_texts_og.utils.prompt_templates import CHUNK_GENERATION_PROMPT, get_variation_hint
from generated_texts_og.utils.story_contract_parser import parse_initial_prompt, format_contract_for_prompt
from generated_texts_og.utils.prompt_sanitizer import strip_chat_prefixes

# tokenizer + metrics (root)
from tokenizer import Tokenizer
from metrics import CoherenceMetrics

# ------------------------------------------------------------
def slugify(text):
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


# =============================================================
# GENERATOR
# =============================================================
class LongBookGenerator:

    def __init__(
        self,
        model_name="mistral",
        device="cuda",
        episodic_interval=5,
        max_chunk_tokens=2048,
        temperature_range=(1.0, 1.4),  # Higher for more varied, creative prose
        top_p=0.9,

        n_attempts=5,
        entropy_threshold=None,
        ppl_threshold=None,
        disable_gating=False,
        disable_self_conditioning=False,
        restart_every=None,
        repetition_penalty=1.1, # Default 1.1
    ):
        # Load config
        with open("generated_texts_og/config/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        if model_name not in config["models"]:
            raise ValueError(f"Model {model_name} not found in config")
            
        model_config = config["models"][model_name]
        
        self.model_config = model_config # Keep for metadata
        
        self.episodic_interval = episodic_interval
        self.max_chunk_tokens = max_chunk_tokens
        self.temperature_range = temperature_range
        self.top_p = top_p
        self.n_attempts = n_attempts
        self.restart_every = restart_every
        self.disable_gating = disable_gating
        self.disable_self_conditioning = disable_self_conditioning
        self.repetition_penalty = repetition_penalty
        
        # Metadata Fix: Update config to reflect runtime overrides
        self.model_config["max_tokens"] = self.max_chunk_tokens
        
        # Load and potentially override thresholds
        from generated_texts_og.utils.coherence_decision import _DEFAULT_CFG
        self.thresholds = _DEFAULT_CFG.copy()
        if entropy_threshold is not None:
             print(f"Override: Setting entropy_min to {entropy_threshold}")
             self.thresholds['entropy_min'] = float(entropy_threshold)
             
        if ppl_threshold is not None:
             print(f"Override: Setting ppl_min to {ppl_threshold}")
             self.thresholds['ppl_min'] = float(ppl_threshold)
        
        # Initialize Model via Factory
        self.model = ModelFactory.create_model(model_config)
        self.device = self.model.device

        # Memory modules

        # Override config context_window if max_chunk_tokens (CLI) implies a larger requirement
        # We estimate 1 token ~= 4 chars
        config_window = model_config.get("context_window", 8000)
        implied_window = self.max_chunk_tokens * 4 
        context_window = max(config_window, implied_window)
        
        print(f"DEBUG: Initializing LocalContext with window={context_window} chars (Config: {config_window}, Implied: {implied_window})")
        self.local_context = LocalContext(max_tokens=context_window)
        self.summary_manager = SummaryManager(episodic_interval=episodic_interval)
        self.graph_memory = GraphMemory()
        self.semantic_memory = SemanticMemory()
        
        # REMOVED: CompressionTransformer and Projection Layer
        # We now use the LLM's native embeddings (Zero-Shot)

        # Serializer
        self.serializer = SmartSerializer(os.path.dirname(__file__))

        # LZ State (for incremental complexity calculation)
        self.lz_state = None
        self.retrieval_vectors = []

    @staticmethod
    def _event_label_from_summary(item, max_chars=96):
        if isinstance(item, dict):
            label = item.get("details", item.get("description", item.get("event", "")))
            if isinstance(label, list):
                label = " ".join(str(x) for x in label)
        elif isinstance(item, list):
            label = " ".join(str(x) for x in item)
        else:
            label = str(item) if item else ""

        label = " ".join(str(label).split()).strip()
        if len(label) > max_chars:
            label = label[:max_chars].rsplit(" ", 1)[0].strip()
        return label

    @staticmethod
    def _normalize_summary_labels(items, field_name):
        labels = []
        if not isinstance(items, list):
            return labels

        for item in items:
            label = None
            if isinstance(item, str):
                label = item
            elif isinstance(item, dict):
                if field_name == "characters":
                    label = item.get("name", item.get("label"))
                elif field_name == "places":
                    label = item.get("name", item.get("location"))
                elif field_name == "objects":
                    label = item.get("name", item.get("label"))
                elif field_name == "concepts":
                    label = item.get("concept", item.get("name"))
            elif item:
                label = str(item)

            label = " ".join(str(label).split()).strip() if label else ""
            if label:
                labels.append(label)
        return labels

    @staticmethod
    def _make_retrieval_excerpt(text, max_chars=900):
        if not text:
            return ""

        paragraphs = [p.strip() for p in str(text).split("\n\n") if p.strip()]
        excerpt_parts = []
        total_chars = 0
        for paragraph in paragraphs:
            excerpt_parts.append(paragraph)
            total_chars += len(paragraph)
            if total_chars >= max_chars:
                break

        excerpt = "\n\n".join(excerpt_parts) if excerpt_parts else " ".join(str(text).split())
        excerpt = " ".join(excerpt.split())
        if len(excerpt) <= max_chars:
            return excerpt

        clipped = excerpt[:max_chars].rsplit(" ", 1)[0].strip()
        if not clipped:
            clipped = excerpt[:max_chars].strip()
        return clipped

    def _build_retrieval_query_text(self, current_context, lead_in_chars=600):
        context_text = str(current_context or "").strip()
        last_chunk_text = str(getattr(self, "last_chunk_text", "") or "").strip()

        if not last_chunk_text:
            return context_text

        prefix = ""
        if context_text.endswith(last_chunk_text):
            prefix = context_text[:-len(last_chunk_text)].rstrip()
        else:
            marker = context_text.rfind(last_chunk_text)
            if marker >= 0:
                prefix = context_text[:marker].rstrip()

        lead_in = prefix[-lead_in_chars:].strip() if prefix else ""
        if lead_in:
            return f"{lead_in}\n\n{last_chunk_text}"

        return last_chunk_text or context_text

    @staticmethod
    def _clip_text_start(text, max_chars):
        text = " ".join(str(text).split())
        if len(text) <= max_chars:
            return text

        clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
        if not clipped:
            clipped = text[:max_chars].strip()
        return clipped + " ..."

    @staticmethod
    def _clip_text_end(text, max_chars):
        text = " ".join(str(text).split())
        if len(text) <= max_chars:
            return text

        clipped = text[-max_chars:].lstrip()
        parts = clipped.split(" ", 1)
        if len(parts) == 2:
            clipped = parts[1].strip()
        return "... " + clipped

    def _load_archived_chunk_text(self, chunk_id):
        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            return ""

        cached = self.retrieval_text_cache.get(chunk_id)
        if cached:
            return cached

        debug_root = getattr(self.serializer, "debug_root", None)
        if not debug_root or not os.path.isdir(debug_root):
            return ""

        prefix = f"chunk_{chunk_id:03d}_attempt_"
        best_path = None
        best_attempt = -1
        for name in os.listdir(debug_root):
            if not name.startswith(prefix) or not name.endswith(".txt") or "_prompt" in name:
                continue
            match = re.search(r"attempt_(\d+)\.txt$", name)
            if not match:
                continue
            attempt_id = int(match.group(1))
            if attempt_id > best_attempt:
                best_attempt = attempt_id
                best_path = os.path.join(debug_root, name)

        if not best_path:
            return ""

        try:
            with open(best_path, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                payload = json.loads(content)
                text = str(payload.get("new_text") or payload.get("full_output") or "").strip()
            except json.JSONDecodeError:
                text = content.strip()
        except Exception:
            return ""

        if text:
            self.retrieval_text_cache[chunk_id] = text
        return text

    def _build_retrieval_prompt_excerpt(self, text):
        text = str(text or "").strip()
        if not text:
            return ""

        text_len = len(text)
        quarter = max(1, text_len // 4)
        head_end = quarter
        tail_start = max(0, text_len - quarter)

        if head_end >= tail_start:
            return text

        head_text = text[:head_end].rstrip()
        tail_text = text[tail_start:].lstrip()
        if not head_text or not tail_text or head_text == tail_text:
            return text

        return f"{head_text}\n\n[...]\n\n{tail_text}"

    def _select_retrieved_chunks(self, current_vector, top_k=2, min_similarity=0.22, recent_exclusion=3):
        records = getattr(self.summary_manager, "retrieval_chunks", [])
        debug = {
            "min_similarity": float(min_similarity),
            "recent_exclusion": int(recent_exclusion),
            "top_k": int(top_k),
            "archive_size": len(records),
            "eligible_candidate_count": 0,
            "threshold_candidate_count": 0,
            "mode": "none",
            "used_fallback": False,
            "selected": [],
        }
        if current_vector is None:
            debug["reason"] = "missing_current_vector"
            return [], debug
        if not records or not self.retrieval_vectors:
            debug["reason"] = "empty_retrieval_archive"
            return [], debug

        total = min(len(records), len(self.retrieval_vectors))
        known_chunk_ids = []
        for record in records[:total]:
            try:
                known_chunk_ids.append(int(record.get("chunk_id")))
            except (TypeError, ValueError):
                continue
        latest_chunk_id = max(known_chunk_ids) if known_chunk_ids else None

        candidates = []
        for idx in range(total):
            record = records[idx]
            vector = self.retrieval_vectors[idx]
            if vector is None:
                continue

            try:
                chunk_id = int(record.get("chunk_id"))
            except (TypeError, ValueError):
                chunk_id = None

            if latest_chunk_id is not None and chunk_id is not None and chunk_id > (latest_chunk_id - recent_exclusion):
                continue
            if latest_chunk_id is None and idx >= max(0, total - recent_exclusion):
                continue

            similarity = MemorySummarizer._cosine_similarity(current_vector, vector)
            candidates.append((similarity, idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        debug["eligible_candidate_count"] = len(candidates)

        threshold_candidates = [item for item in candidates if item[0] >= min_similarity]
        debug["threshold_candidate_count"] = len(threshold_candidates)

        chosen_candidates = threshold_candidates
        if threshold_candidates:
            debug["mode"] = "threshold"
        elif candidates:
            chosen_candidates = candidates
            debug["mode"] = "fallback"
            debug["used_fallback"] = True
        else:
            debug["reason"] = "no_eligible_candidates_after_recent_exclusion"
            return [], debug

        selected = []
        seen_chunk_ids = set()
        for similarity, idx in chosen_candidates:
            record = records[idx]
            chunk_id = record.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue

            archived_text = self._load_archived_chunk_text(chunk_id)
            prompt_excerpt = self._build_retrieval_prompt_excerpt(
                archived_text or record.get("excerpt", "")
            )

            selected.append({
                "chunk_id": chunk_id,
                "excerpt": record.get("excerpt", ""),
                "prompt_excerpt": prompt_excerpt or record.get("excerpt", ""),
                "similarity": round(float(similarity), 4),
            })
            seen_chunk_ids.add(chunk_id)
            if len(selected) >= top_k:
                break

        debug["selected"] = [
            {
                "chunk_id": item.get("chunk_id"),
                "similarity": item.get("similarity"),
            }
            for item in selected
        ]
        if not selected:
            debug["mode"] = "none"
            debug["used_fallback"] = False
            debug["reason"] = "no_unique_selection"

        return selected, debug

    def _rebuild_retrieval_index(self):
        self.retrieval_vectors = []
        retrieval_chunks = getattr(self.summary_manager, "retrieval_chunks", [])
        if not retrieval_chunks:
            return

        print(f"   ⏳ Rebuilding raw-chunk retrieval index from {len(retrieval_chunks)} excerpts...")
        for record in retrieval_chunks:
            excerpt = str(record.get("excerpt", "")).strip()
            if not excerpt:
                self.retrieval_vectors.append(None)
                continue

            try:
                vec = self.model.get_embedding(excerpt)
                self.retrieval_vectors.append(vec[0].detach().cpu().numpy().astype(float))
            except Exception as e:
                print(f"   ⚠️ Retrieval embedding rebuild failed: {e}")
                self.retrieval_vectors.append(None)
        print("   ✓ Retrieval index restored.")

    def _update_graph_from_summary(self, summary):
        if not isinstance(summary, dict):
            return

        self.graph_memory.age_all()

        characters = self._normalize_summary_labels(summary.get("characters", []), "characters")
        places = self._normalize_summary_labels(summary.get("places", []), "places")
        objects = self._normalize_summary_labels(summary.get("objects", []), "objects")
        concepts = self._normalize_summary_labels(summary.get("concepts", []), "concepts")
        raw_events = summary.get("main_events", [])
        events = [self._event_label_from_summary(item) for item in raw_events]
        events = [evt for evt in events if evt]

        char_ids = [self.graph_memory.register_node(label, "character") for label in characters]
        place_ids = [self.graph_memory.register_node(label, "place") for label in places]
        object_ids = [self.graph_memory.register_node(label, "object") for label in objects]
        concept_ids = [self.graph_memory.register_node(label, "concept") for label in concepts]
        event_ids = [self.graph_memory.register_node(label, "event") for label in events[:2]]

        for i, src_id in enumerate(char_ids):
            for dst_id in char_ids[i + 1:]:
                self.graph_memory.register_edge(src_id, dst_id, "co_occurs_with")
                self.graph_memory.register_edge(dst_id, src_id, "co_occurs_with")

        for char_id in char_ids:
            for place_id in place_ids[:2]:
                self.graph_memory.register_edge(char_id, place_id, "appears_in")

        for char_id in char_ids:
            for object_id in object_ids[:2]:
                self.graph_memory.register_edge(char_id, object_id, "mentions")

        for event_id in event_ids:
            for char_id in char_ids[:3]:
                self.graph_memory.register_edge(event_id, char_id, "involves")
            for place_id in place_ids[:2]:
                self.graph_memory.register_edge(event_id, place_id, "occurs_in")
            for concept_id in concept_ids[:2]:
                self.graph_memory.register_edge(event_id, concept_id, "centers_on")

        self.graph_memory.prune()

    # ------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------
    def generate_book(self, initial_prompt, n_chunks, target_tokens=None, resume_from_dir=None, run_id=None, prompt_filename="Manual"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check for persistent run directory
        if run_id:
            msg = f"Checking for persistent run directory: {run_id}"
            run_dir = os.path.join(self.serializer.prompts_dir, run_id)
            if os.path.exists(run_dir):
                print(f"{msg} -> Found. Resuming.")
                resume_from_dir = run_dir
            else:
                print(f"{msg} -> Not found. Starting fresh in {run_id}.")
                # will be handled in fresh start block

        self.prompt_filename = prompt_filename

        
        # Track full text and tokens for global metrics
        self.full_text = ""
        self.full_tokens = [] 
        self.total_chunks = n_chunks  # Default estimate
        self.target_tokens = target_tokens # Store target tokens logic 
        
        # Reset LZ State
        self.lz_state = None
        self.retrieval_vectors = []
        self.last_chunk_text = ""
        self.retrieval_text_cache = {}

        start_chunk_id = 1
        
        # ------------------------------------------------------------------
        # RESUME LOGIC
        # ------------------------------------------------------------------
        if resume_from_dir and os.path.exists(resume_from_dir):
            print(f"🔄 Resuming from: {resume_from_dir}")
            from generated_texts_og.utils.resume_helper import find_latest_chunk_id, load_memory_snapshot, reconstruct_text_history
            
            last_id = find_latest_chunk_id(resume_from_dir)
            mem_snap = None
            if last_id > 0:
                print(f"   Found checkpoint at chunk {last_id}")
                
                # SAFE RESUME LOGIC: Discard the last chunk to ensure clean state
                print(f"   🛡️ SAFE RESUME: Discarding last chunk ({last_id}) to prevent corruption.")
                last_id = max(0, last_id - 1)
                print(f"   -> Resuming from Chunk {last_id}")

                if last_id == 0:
                     print("   (Restarting from scratch effectively)")
                     # Fall through to fresh start logic? 
                     # The code below relies on loading a snapshot. 
                     # If last_id is 0, we treat it as fresh start.
                     start_chunk_id = 1
                else:
                    # 1. Restore Memory
                    mem_snap = load_memory_snapshot(resume_from_dir, last_id)
                if mem_snap:
                    self.summary_manager.chunks = mem_snap.get("recent_chunks", [])
                    self.summary_manager.episodes = mem_snap.get("episodes", [])
                    self.summary_manager.global_summary = mem_snap.get("global_summary", "")
                    self.summary_manager.retrieval_chunks = mem_snap.get("retrieval_chunks", [])
                    self.summary_manager.pending_episode_source = None
                    
                    # Try to load tokens.npy for exact state restoration
                    tokens_path = os.path.join(resume_from_dir, f"memory_{last_id:03d}", "tokens.npy")
                    if os.path.exists(tokens_path):
                        try:
                            self.full_tokens = list(np.load(tokens_path))
                            print(f"   ✓ RESTORED EXACT TOKEN STATE from tokens.npy ({len(self.full_tokens)} tokens)")
                            
                            # Restore LZ State (Requires one-time compute)
                            if len(self.full_tokens) > 0:
                                 print("   ⏳ Re-hydrating LZ Complexity state (this may take a moment)...")
                                 # We compute from scratch ONCE to get the state
                                 m = CoherenceMetrics(self.full_tokens).compute_all()
                                 if 'lz_complexity_count' in m:
                                     # Correctly restore state including l_ptr
                                     self.lz_state = {
                                         'c': m['lz_complexity_count'],
                                         'l_ptr': m.get('lz_complexity_l', len(self.full_tokens))
                                     }
                                     print(f"   ✓ LZ State restored: c={self.lz_state['c']}, l={self.lz_state['l_ptr']}")
                                 else:
                                     print("   ⚠️ LZ State could not be restored (metrics missing count).")

                        except Exception as e:
                            print(f"   ⚠️ Failed to load tokens.npy: {e}")
                            self.full_tokens = []
                    else:
                        self.full_tokens = []

                    # Restore semantic memory if possible (serialized as list)
                    sem_data = mem_snap.get("semantic_memory", [])
                    # SemanticMemory structure reconstruction (simplified)
                    if isinstance(sem_data, dict):
                        raw_vecs = sem_data.get("vectors", [])
                        self.summary_manager.semantic_memory.memory = [np.array(v) for v in raw_vecs]
                        self.summary_manager.semantic_memory.dim = sem_data.get("dim", len(raw_vecs[0]) if raw_vecs else 64)
                    elif isinstance(sem_data, list):
                        # Fallback for legacy format if any
                        self.summary_manager.semantic_memory.memory = [np.array(v) for v in sem_data]

                    graph_data = mem_snap.get("graph_memory")
                    if graph_data:
                        self.graph_memory.restore_from_serializable(graph_data)
                    elif self.summary_manager.chunks:
                        for summary in self.summary_manager.chunks:
                            self._update_graph_from_summary(summary)
                    print("   ✓ Memory restored.")

                # 2. Restore Text History
                restored_full_text, last_chunk_text, text_parts = reconstruct_text_history(resume_from_dir, last_id)
                self.full_text = restored_full_text
                self.last_chunk_text = last_chunk_text or ""
                
                # 3. Restore Local Context (from last chunk purely)
                if last_chunk_text:
                    self.local_context.reset(last_chunk_text)
                
                # 4. Re-tokenize history (Only if tokens.npy was missing)
                if not self.full_tokens:
                    print(f"   Re-tokenizing {len(text_parts)} chunks from history...")
                    self.full_tokens = []
                    enc = Tokenizer(kind="gpt4")
                    
                    # Optimization: Remove local import that shadows global
                    # from tqdm import tqdm
                    for part in tqdm(text_parts, desc="Tokenizing History"):
                         tokens = enc.encode(part)
                         self.full_tokens.extend(tokens)
                    print(f"   ✓ Tokenization complete: {len(self.full_tokens)} tokens.")
                else:
                    print(f"   ✓ Skipping re-tokenization (loaded {len(self.full_tokens)} tokens).")

                self._rebuild_retrieval_index()
                print(f"   ✓ History ready ({len(self.full_tokens)} tokens)")

                # 5. Set pointers
                start_chunk_id = last_id + 1
                
                 # 6. Reuse the SAME directory for new outputs
                # We need to tell serializer to use this dir.
                
                # Attempt to extract timestamp from metadata first (more reliable for run_id folders)
                old_ts = None
                meta_path = os.path.join(resume_from_dir, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        try:
                            meta_loaded = json.load(f)
                            old_ts = meta_loaded.get("timestamp")
                        except:
                            pass
                
                if not old_ts:
                    # Fallback to directory name parsing (legacy behavior)
                    dirname = os.path.basename(resume_from_dir.rstrip("/"))
                    parts = dirname.split("_")
                    if len(parts) >= 2 and parts[0].isdigit():
                        old_ts = parts[0] + "_" + parts[1]
                    else:
                        old_ts = timestamp # Fallback to current if we absolutely can't find it

                self.serializer._pending_timestamp = old_ts
                self.serializer._pending_temp_dir = resume_from_dir
                self.serializer.debug_root = resume_from_dir
                
                # Reuse metadata
                # (We assume metadata.json exists and is valid)
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        self.story_contract = meta.get("story_contract", "")
                        self.story_contract_compact = meta.get("story_contract_compact", "")
                    # narrative_seed might be in full_text already
            else:
                print("   No valid checkpoints found. Starting fresh.")
        
        # ------------------------------------------------------------------
        # FRESH START LOGIC (only if not resumed)
        # ------------------------------------------------------------------
        if start_chunk_id == 1:
            # Parse initial prompt into narrative seed and story contract
            narrative_seed, story_contract = parse_initial_prompt(initial_prompt.strip())
            self.story_contract = story_contract # Store for chunk generation
            
            # Format contract for compact injection into chunk prompts
            self.story_contract_compact = format_contract_for_prompt(story_contract, compact=True)

            # Phase 1: Create timestamp-only folder (or run_id folder)
            self.serializer.init_temp_book_folder(timestamp, custom_folder_name=run_id)

            # Save initial prompt to file
            prompt_path = os.path.join(self.serializer.debug_root, "initial_prompt.md")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(initial_prompt.strip())

            # Save initial metadata (include both parts)
            metadata = {
                "timestamp": timestamp,
                "model_config": self.model_config,
                "prompt_filename": prompt_filename, # Added
                "initial_prompt_full": initial_prompt.strip(),
                "narrative_seed": narrative_seed,
                "story_contract": story_contract,
                "story_contract_compact": self.story_contract_compact,
                "broader_genre": "Pending",
                "specific_genre": "Pending",
                "title": "Pending",
                "title_slug": "pending",
                "max_chunk_tokens": self.max_chunk_tokens,
                "temperature_range": self.temperature_range,
                "n_attempts": self.n_attempts,
                "repetition_penalty": self.repetition_penalty,
                "disable_self_conditioning": self.disable_self_conditioning,
                "debug_folder": self.serializer.debug_root,
                "compression_method": "llm_native_hidden_state"
            }
            self.serializer.save_metadata(metadata)

            # Reset memories with ONLY narrative seed (not full prompt)
            self.local_context.reset(narrative_seed)
            self.summary_manager.reset()
            self.graph_memory = GraphMemory()
            self.semantic_memory = SemanticMemory()
            
            # Track full text for global metrics
            self.full_text = narrative_seed + "\n\n"
            self.full_tokens = Tokenizer(kind="gpt4").encode(self.full_text)

        # Text buffer (start with just the narrative seed or restored full text)
        final_text = self.full_text

        # Generate all chunks
        chapter_num = 1 + ((start_chunk_id - 1) // self.episodic_interval)
        
        # Calculate initial estimate for display
        if self.target_tokens:
             est_chunks = int(self.target_tokens / (self.max_chunk_tokens * 0.8))
             initial_total = max(est_chunks, start_chunk_id + 10)
        else:
             initial_total = n_chunks

        # Use a large range so we don't stop early if chunks are short
        # pbar = tqdm(range(start_chunk_id, 99999), desc="LongBook", total=initial_total)
        # for chunk_id in pbar:
        for chunk_id in range(start_chunk_id, 99999):
            
            # Dynamic Termination Check
            if self.target_tokens:
                 if len(self.full_tokens) >= self.target_tokens:
                     print(f"\n✅ Target tokens reached: {len(self.full_tokens)} >= {self.target_tokens}")
                     # Ensure we end gracefully
                     break
            elif chunk_id > n_chunks:
                 # Standard chunk limit
                 break
            
            # Add Chapter Header if start of new episode (or first chunk)
            if chunk_id == 1 or (chunk_id - 1) % self.episodic_interval == 0:
                if chunk_id > 1:
                    chapter_num += 1
                header = f"\n\n## Chapter {chapter_num}\n\n"
                final_text += header
                # Also inject into context so model knows (optional, but good for coherence)
                self.local_context.update_context(header)

            tgt_str = f"{self.target_tokens:,}" if self.target_tokens else 'N/A'
            curr_str = f"{len(self.full_tokens):,}"
            print(f"\nGenerating Chunk {chunk_id}... (Tokens: {curr_str}/{tgt_str})")
            new_chunk = self._generate_one_chunk(chunk_id)
            
            # Append to text
            final_text += new_chunk + "\n\n"
            self.full_text += new_chunk + "\n\n"
            self.local_context.update_context(new_chunk)
            
            # Append to tokens (Optimized)
            new_toks = Tokenizer(kind="gpt4").encode(new_chunk + "\n\n")
            self.full_tokens.extend(new_toks)

            # Dynamic Progress Update (Print)
            if self.target_tokens:
                 current_total = len(self.full_tokens)
                 percentage = (current_total / self.target_tokens) * 100
                 print(f"   -> Progress: {current_total:,} / {self.target_tokens:,} tokens ({percentage:.2f}%)")


            
            # Generate episodic summary every N chunks
            if chunk_id % self.episodic_interval == 0 and self.summary_manager.pending_episode_source:
                self._generate_episodic_summary()
                
            # AGGRESSIVE MEMORY CLEANUP
         # Explicit Cleanup after every chunk (Aggressive VRAM Hygiene)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Check for Auto-Restart (Fragmentation Mitigation)
        chunks_generated_this_session = chunk_id - start_chunk_id + 1
        if self.restart_every and chunks_generated_this_session >= self.restart_every:
             print(f"\n🔄 Auto-Restart Triggered after {chunks_generated_this_session} chunks to clear VRAM fragmentation.")
             print(f"Exiting with code 42 (Resume needed). Next Chunk: {chunk_id + 1}")
             sys.exit(42)

        # Phase 2: Generate title/genre from final memory
        print("\n=== Generating title and genre from final memory ===")
        
        # Prepare memory for title generation
        memory_data = self.summary_manager.to_serializable()
        
        # Inject Initial Prompt Use (Fixing Recency Bias)
        if hasattr(self, 'initial_prompt') and self.initial_prompt:
             memory_data["initial_prompt"] = self.initial_prompt
        
        if hasattr(self, 'prompt_filename') and self.prompt_filename:
             memory_data["prompt_filename"] = self.prompt_filename

        # Use robust extraction
        broader, specific, title = generate_title_and_genre(memory_data, self.model)
        title_slug = slugify(title)

        # Phase 3: Rename folder with title and genre
        final_folder = self.serializer.finalize_book_folder(title_slug, broader)

        # Resave metadata
        try:
            with open(os.path.join(final_folder, "metadata.json"), 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
            
        metadata.update({
            "broader_genre": broader,
            "specific_genre": specific,
            "title": title,
            "title_slug": title_slug,
            "debug_folder": final_folder,
            "total_tokens": len(self.full_tokens),
        })
        self.serializer.save_metadata(metadata)

        # Add title to final text
        final_text = f"# {title}\n\n## {broader} - {specific}\n\n" + final_text

        # Save final text
        return self.serializer.save_book_text(
            self.model.model_name.replace("/", "_"), title_slug, broader, timestamp, final_text
        )

    # ------------------------------------------------------------
    # CHUNK GENERATION
    # ------------------------------------------------------------
    def _generate_one_chunk(self, chunk_id):

        # 1. Convert memory to natural language summary
        current_semantic_vector_list = None
        current_semantic_tensor = None
        retrieval_debug = {"mode": "disabled", "reason": "self_conditioning_disabled"} if self.disable_self_conditioning else {"mode": "none"}
        memory_summary = ""
        current_context = self.local_context.get_context()

        if not self.disable_self_conditioning:
            memory_data = self.summary_manager.to_serializable()
            memory_data["recent_chunks"] = memory_data.get("recent_chunks", [])
            memory_data["global_summary"] = memory_data.get("global_summary", "")
            memory_data["graph_memory"] = self.graph_memory.to_serializable()

            # 1.5. Compute current context semantic vector using LLM Native Embedding
            query_text = self._build_retrieval_query_text(current_context)
            if query_text and len(query_text) > 100:

                # Get LLM's internal representation (Zero-Shot)
                # Returns (1, hidden_size) e.g. (1, 4096)
                vec = self.model.get_embedding(query_text)

                current_semantic_tensor = vec # Keep tensor for injection
                current_semantic_vector_list = vec[0].cpu().tolist() # List for summarizer

            retrieved_chunks, retrieval_debug = self._select_retrieved_chunks(current_semantic_vector_list)
            if retrieved_chunks:
                memory_data["retrieved_chunks"] = retrieved_chunks

            memory_summary = MemorySummarizer.summarize_for_generation(
                memory_data,
                current_semantic_vector=current_semantic_vector_list
            )
        
        # 2. Build natural language prompt
        # Use minimal context (last ~2000 chars) to give the model immediate continuity
        prev_text = self.local_context.get_minimal_context()
        
        # Calculate target word count based on max_tokens (approx 0.75 words per token)
        # 2048 tokens -> ~1500 words
        # 4096 tokens -> ~3000 words
        target_words = int(self.max_chunk_tokens * 0.75)
        # Determine identity based on prompt_filename or metadata
        # (Assuming prompts filenames contain keywords like 'science', 'philosophy', 'technical')
        
        identity_prompt = "You are a novelist continuing a story." # Default
        
        # Check metadata or filename for non-fiction keywords
        pf = self.prompt_filename.lower() if hasattr(self, 'prompt_filename') and self.prompt_filename else ""
        
        if "science" in pf or "physics" in pf or "biology" in pf:
            identity_prompt = "You are an expert science writer and educator."
        elif "philosophy" in pf:
            identity_prompt = "You are a philosopher writing a comprehensive treatise."
        elif "history" in pf or "historical" in pf:
             # Historical fiction is still a 'novelist' task usually, but maybe 'historian' if non-fiction
             if "library" in pf: # historical_library.md
                 identity_prompt = "You are a historical novelist writing an immersive period piece."
        elif "technical" in pf:
             identity_prompt = "You are a technical writer creating a detailed reference guide."
        elif "biography" in pf:
             identity_prompt = "You are a meticulous biographer recounting a real life."
             
        length_str = f"{target_words}-{target_words + 500}"
        
        base_prompt = CHUNK_GENERATION_PROMPT.format(
            identity_prompt=identity_prompt,
            story_contract=self.story_contract,
            memory_summary=memory_summary,
            previous_text=prev_text,
            variation_hint=get_variation_hint(chunk_id),
            target_length=length_str
        )

        last_output = ""

        # ------------------------------------------------------------------
        # ENDING LOGIC INJECTION
        # ------------------------------------------------------------------
        ending_instruction = ""
        
        if self.target_tokens:
            # DYNAMIC TOKEN TARGET LOGIC
            current_tokens = len(self.full_tokens)
            remaining_tokens = self.target_tokens - current_tokens
            
            # Estimate remaining chunks for "Finale" logic (assuming 1k tokens/chunk for safety)
            # This is just for prompt phrasing "X chunks remaining"

            # Handle potential zero division if full_tokens is empty (e.g. at very start)
            avg_toks = len(self.full_tokens) / max(1, chunk_id)
            if avg_toks <= 0:
                avg_toks = self.max_chunk_tokens # Fallback to max_chunk_tokens as estimate
            
            est_remaining_chunks = max(1, int(remaining_tokens / avg_toks))

            if remaining_tokens <= 0: # Should be caught by loop break, but safe to handle
                 ending_instruction = "\n[SYSTEM: THIS IS THE FINAL CHUNK. You MUST write 'THE END' at the conclusion of this text. Resolve all remaining tension and bring the story to a satisfying close.]\n"
            elif remaining_tokens < 5000: # Last ~5 chunks
                 ending_instruction = f"\n[SYSTEM: This is the FINALE (~{est_remaining_chunks} chunks remaining). Write the climax and conclusion. Provide a satisfying resolution to the story.]\n"
            elif remaining_tokens < 100000: # Wrapping Up Phase (Last 100k)
                 ending_instruction = f"\n[SYSTEM: You are approaching the end of the book (~{remaining_tokens} tokens remaining). Begin resolving major plot lines, closing arcs, and moving characters towards the finale. Do not introduce new subplots.]\n"

        else:
            # ORIGINAL CHUNK-BASED LOGIC
            # Determine ending zone size (Token-Aware Cap):
            hard_cap_tokens = 100000
            total_est_tokens = self.total_chunks * self.max_chunk_tokens
            
            wrap_duration_tokens = min(hard_cap_tokens, int(total_est_tokens * 0.10))
            # Convert to chunks
            start_wrapping_threshold = max(2, int(wrap_duration_tokens / self.max_chunk_tokens))
            
            # Ensure we don't start wrapping immediately for tiny tests
            if self.total_chunks < 10:
                 start_wrapping_threshold = 2
                 
            remaining = self.total_chunks - chunk_id
            
            if remaining == 0:
                ending_instruction = "\n[SYSTEM: THIS IS THE FINAL CHUNK. You MUST write 'THE END' at the conclusion of this text. Resolve all remaining tension and bring the story to a satisfying close.]\n"
            elif remaining < 5:
                ending_instruction = f"\n[SYSTEM: This is the FINALE ({remaining} chunks remaining). Write the climax and conclusion. Provide a satisfying resolution to the story.]\n"
            elif remaining < start_wrapping_threshold:
                ending_instruction = f"\n[SYSTEM: You are approaching the end of the book ({remaining} chunks remaining). Begin resolving major plot lines, closing arcs, and moving characters towards the finale. Do not introduce new subplots.]\n"
        
        if ending_instruction:
            base_prompt += ending_instruction

        from generated_texts_og.utils.coherence_decision import is_chunk_acceptable, check_start_repetition, calculate_metric_distance
        
        failed_attempts = []
        accumulated_metrics = None # Initialize to avoid UnboundLocalError

        for attempt in range(1, self.n_attempts + 1):
            # Log prompt before execution
            self.serializer.save_debug_prompt(
                f"chunk_{chunk_id:03d}_attempt_{attempt}_prompt",
                base_prompt
            )

            # Temperature schedule
            T_low, T_high = self.temperature_range
            T = T_low + (attempt - 1) / max(1, self.n_attempts - 1) * (T_high - T_low)
            
            # Temporarily set model temperature
            self.model.temperature = T

            # Dynamic Sampling Parameters for High Attempts
            current_top_p = self.top_p
            current_rep_penalty = self.repetition_penalty
            
            # If failing (attempt 3+), relax constraints to allow higher entropy
            if attempt >= 3:
                current_top_p = 0.98  # Open up the tail
                current_rep_penalty = max(self.repetition_penalty, 1.15) # Force vocabulary diversity
            if attempt >= 4:
                current_top_p = 1.0   # Full random tail
            
            # Chaos Prompt Injection (Attempts 4 & 5)
            # Break "Entropy Inertia" by forcing a narrative discontinuity
            prompt_to_use = base_prompt
            if attempt >= 4:
                chaos_instruction = "\n[SYSTEM: FAILURE IMMINENT. INCREASE ENTROPY. WRITE UNPREDICTABLY. USE COMPLEX, DISJOINTED IMAGERY. TAKE A SHARP NARRATIVE TURN.]\n"
                prompt_to_use = base_prompt + chaos_instruction

            # Prepare Soft Prompt (Vector Injection)
            prefix_embeds = None
            if (not self.disable_self_conditioning) and current_semantic_tensor is not None:
                # Use the native embedding directly as a prefix
                # Shape is already (1, hidden_size), need (1, 1, hidden_size)
                prefix_embeds = current_semantic_tensor.unsqueeze(1)

            # MEMORY DEBUG
            if chunk_id % 5 == 0 and attempt == 1:
                print(f"  [Mem] Pre-Gen: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")

            try:
                full_output = self.model.generate(
                    prompt_to_use, 
                    max_new_tokens=self.max_chunk_tokens,
                    prefix_embeds=prefix_embeds,
                    top_p=current_top_p,
                    repetition_penalty=current_rep_penalty
                )
            except Exception as e:
                print(f"Generation error attempt {attempt}: {e}")
                # Aggressive cleanup on error
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            # Extract text (still useful to clean up potential artifacts)
            new_text = extract_new_text(self.local_context.get_context(), full_output, prompt=base_prompt)
            
            # --- NEW: LIGHTWEIGHT SANITIZATION (REGEX) ---
            if new_text:
                new_text = strip_chat_prefixes(new_text)
                
            last_output = new_text

            # Skip this attempt if extraction resulted in empty text
            if not new_text or len(new_text.strip()) == 0:
                print(f"Warning: Chunk {chunk_id} attempt {attempt} produced empty text after filtering. Retrying...")
                continue
            
            # Sanitization check for contamination
            from generated_texts_og.utils.prompt_sanitizer import quick_contamination_check, sanitize_chunk
            if quick_contamination_check(new_text):
                print(f"  ⚠️  Contamination detected in chunk {chunk_id} attempt {attempt}, sanitizing...")
                new_text = sanitize_chunk(new_text, self.story_contract_compact, self.model)
                last_output = new_text
                
                # Re-check if sanitization resulted in empty text
                if not new_text or len(new_text.strip()) == 0:
                    print(f"  ❌ Sanitization produced empty text. Retrying...")
                    continue

            # MEMORY DEBUG
            if chunk_id % 5 == 0 and attempt == 1:
                print(f"  [Mem] Post-Gen: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")

            try:
                # Tokenize + metrics (Local)
                toks = Tokenizer(kind="gpt4").encode(new_text)
                
                # Additional safety: check if tokenization resulted in empty token list
                if len(toks) == 0:
                    print(f"Warning: Chunk {chunk_id} attempt {attempt} produced 0 tokens. Retrying...")
                    continue
                    
                chunk_metrics = CoherenceMetrics(toks).compute_all()

                # Cleanup local metrics tensors immediately?
                # CoherenceMetrics uses 'compute_all' which returns simpler types (float), but check for lingering tensors.
                torch.cuda.empty_cache()

                # MEMORY DEBUG
                if chunk_id % 5 == 0 and attempt == 1:
                    print(f"  [Mem] Pre-Global: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")

                # Tokenize + metrics (Global/Accumulated)
                # We use the full tokens generated so far + this new chunk
                # This allows us to track Beta and LRD evolution over time
                
                # Optimized: use cached tokens
                if len(self.full_tokens) > 200:  
                    # Avoid re-encoding giant string
                    global_toks = self.full_tokens + toks
                    
                    # Incremental LZ Logic
                    cm = CoherenceMetrics(global_toks)
                    cm.lz_state = self.lz_state
                    accumulated_metrics = cm.compute_all()
                else:
                    # Not enough text yet—fall back to per-chunk metrics
                    accumulated_metrics = chunk_metrics
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠️ OOM during global metric calculation for attempt {attempt}. Fallback to local metrics.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    accumulated_metrics = chunk_metrics
                    # Do not continue; proceed to evaluation using local metrics
                else:
                    raise e
            except Exception as e:
                 print(f"  ⚠️ Error during metrics: {e}")
                 chunk_metrics = {}
                 accumulated_metrics = {}

            # Save full attempt JSON
            attempt_dump = {
                "prompt": base_prompt,
                "temperature": T,
                "retrieval_debug": retrieval_debug,
                "full_output": full_output,
                "new_text": new_text,
                "chunk_metrics": chunk_metrics,
                "accumulated_metrics": accumulated_metrics,
            }
            self.serializer.save_debug_prompt(
                f"chunk_{chunk_id:03d}_attempt_{attempt}",
                json.dumps(attempt_dump, indent=2),
            )

            # Accept?
            from generated_texts_og.utils.coherence_decision import check_start_repetition
            
            # Check for stuttering (repeating end of context)
            if check_start_repetition(new_text, self.local_context.get_context(), n_chars=50):
                # Reject if it just repeats the previous text
                print(f"  ❌ Stutter repetition detected in attempt {attempt}.")
                continue

            # Use accumulated metrics for acceptance to align with coherence thresholds over the running book
            # BUT use local text for repetition checks (passing accumulated text would trigger false positives for repeated words across chapters)
            acceptance_metrics = accumulated_metrics or chunk_metrics
            
            # Calculate distance score (lower is better)
            distance_score = calculate_metric_distance(
                acceptance_metrics, 
                chunk_id=chunk_id, 
                cfg=self.thresholds,
                verbose=(attempt == 1),
                text=new_text
            )
            
            # Store attempt data for potential fallback
            # We must be careful not to store large tensors. 
            # We assume chunk_metrics contain only floats/ints.
            # But we should detach them just in case.
            
            failed_attempts.append({
                "text": new_text, 
                "chunk_metrics": chunk_metrics,
                "distance": distance_score,
                "attempt": attempt
            })
            
            # Explicit cleanup for this iteration
            # if 'accumulated_metrics' in locals(): del accumulated_metrics  # Removed to fix UnboundLocalError
            gc.collect()

            # Only accumulated metrics should be gated, not local ones.
            # We satisfy this by passing acceptance_metrics (accumulated).
            # However, for the 'text' argument which checks for immediate stuttering/n-gram repetition,
            # we MUST use the new_text, otherwise we'd re-scan the whole book for repeats which is wrong.
            if is_chunk_acceptable(acceptance_metrics, text=new_text, chunk_id=chunk_id, cfg=self.thresholds, disable_gating=self.disable_gating):
                self._update_memory(new_text, chunk_metrics, chunk_id, accumulated_metrics=accumulated_metrics)
                
                # Update LZ State
                if 'lz_complexity_count' in acceptance_metrics:
                    # Update State
                    self.lz_state = {
                        'c': acceptance_metrics['lz_complexity_count'],
                        # Use l_ptr if available, else fallback to length (which is incorrect but safe fallback)
                        'l_ptr': acceptance_metrics.get('lz_complexity_l', len(self.full_tokens))
                    }
                
                # Aggressive Memory Cleanup
                del failed_attempts
                if 'chunk_metrics' in locals(): del chunk_metrics
                if 'accumulated_metrics' in locals(): del accumulated_metrics
                gc.collect()
                torch.cuda.empty_cache()
                
                return new_text
            else:
                 m = acceptance_metrics
                 # Format metrics for display with color coding
                 # Use self.thresholds instead of importing _DEFAULT_CFG
                 from generated_texts_og.utils.coherence_decision import GATED_METRICS
                 CFG = self.thresholds
                 
                 def color_metric(key, display_name, val, fmt=".3f"):
                     if key not in GATED_METRICS:
                         return f"{display_name}: {val:{fmt}}"
                     
                     min_k, max_k = None, None
                     if key == "beta": min_k, max_k = "beta_min", "beta_max"
                     elif key == "perplexity": min_k, max_k = "ppl_min", "ppl_max"
                     elif key == "entropy_bits": min_k, max_k = "entropy_min", "entropy_max"
                     elif key == "lz_complexity": min_k, max_k = "lz_min", "lz_max"
                     elif key == "mi_decay_gamma": min_k, max_k = "gamma_min", "gamma_max"
                     elif key == "alpha": min_k, max_k = "alpha_min", "alpha_max"
                     elif key == "H_DMA": min_k, max_k = "H_DMA_min", "H_DMA_max"
                     
                     if not min_k: return f"{display_name}: {val:{fmt}}"
                     
                     low = CFG.get(min_k, -float('inf'))
                     high = CFG.get(max_k, float('inf'))
                     
                     ANSI_RED = "\033[91m"
                     ANSI_GREEN = "\033[92m"
                     ANSI_RESET = "\033[0m"
                     
                     if val < low or val > high:
                         return f"{display_name}: {ANSI_RED}{val:{fmt}}{ANSI_RESET}"
                     else:
                         return f"{display_name}: {ANSI_GREEN}{val:{fmt}}{ANSI_RESET}"

                 metric_parts = []
                 metric_parts.append(color_metric("beta", "Beta", m.get('beta', 0)))
                 metric_parts.append(color_metric("alpha", "Alpha", m.get('alpha', 0)))
                 metric_parts.append(color_metric("perplexity", "PPL", m.get('perplexity', 0), ".0f"))
                 metric_parts.append(color_metric("entropy_bits", "Ent", m.get('entropy_bits', 0), ".2f"))
                 metric_parts.append(color_metric("lz_complexity", "Lz", m.get('lz_complexity', 0), ".4f"))
                 metric_parts.append(color_metric("H_DMA", "H_DMA", m.get('H_DMA', 0), ".2f"))
                 metric_parts.append(color_metric("mi_decay_gamma", "Gam", m.get('mi_decay_gamma', 0), ".4f"))
                 
                 metric_str = ", ".join(metric_parts)
                 if distance_score >= 100.0:
                      print(f"  ❌ Attempt {attempt} rejected. ⚠️ Stutter (Dist: {distance_score:.4f}) | {metric_str}")
                 else:
                      print(f"  ❌ Attempt {attempt} rejected. Dist: {distance_score:.4f} | {metric_str}")

        # Fallback: Select the best attempting (lowest distance)
        # Fallback: Select the best attempting (lowest distance)
        if failed_attempts:
            best_attempt = min(failed_attempts, key=lambda x: x["distance"])
            print(f"⚠️ All attempts failed. Fallback to attempt {best_attempt['attempt']} (Best Distance: {best_attempt['distance']:.4f})")
            # Note: best_attempt['metrics'] is local, not accumulated. 
            # Ideally we'd have the accumulated metrics corresponding to this attempt, but we only calculate it 
            # during the check. We might need to re-calculate or assume it's close enough.
            # However, for saving state, we likely want the accumulated metrics of the ACCEPTED chunk.
            # If we fall back, we accept it. So we should re-calc accumulated here?
            # Or just pass existing 'accumulated_metrics' if available from last loop?
            # Issue: 'accumulated_metrics' var is local to loop. 
            pass
            # For now, pass None if not available, OR better:
            # We must ensure accumulated_metrics is available.
            
            # If we are here, 'accumulated_metrics' might be strictly from the LAST attempt.
            # But the BEST attempt might be an earlier one.
            # So we should ideally recompute or use what we have.
            # For safety, let's just pass what we have (or {}).
            # The UnboundLocalError happens because 'accumulated_metrics' wasn't defined if loop didn't run or all failed early?
            # Actually it IS defined in the loop, but python scoping makes it available. 
            # BUT if all attempts 'continue'd before assigning it (e.g. generation error), it would be unbound.
            
            self._update_memory(best_attempt['text'], best_attempt['chunk_metrics'], chunk_id, accumulated_metrics=accumulated_metrics)
            
            # Aggressive Memory Cleanup
            del failed_attempts
            gc.collect()
            torch.cuda.empty_cache()
            
            return best_attempt['text']
        
        # Absolute fallback if no attempts succeeded (e.g. all empty or stuttering)
        if not last_output or len(last_output.strip()) == 0:
            raise RuntimeError(f"Chunk {chunk_id}: Generation failed after {self.n_attempts} attempts. Aborting run to prevent data corruption.")

        self._update_memory(last_output, {}, chunk_id, accumulated_metrics=accumulated_metrics)
        return last_output

    # ------------------------------------------------------------
    # EPISODIC SUMMARY GENERATION
    # ------------------------------------------------------------
    def _generate_episodic_summary(self):
        """Generate an episodic summary from the last N chunks using the LLM."""
        from .utils.prompt_templates import EPISODIC_SUMMARY_PROMPT
        
        # Get the pending episode chunks
        chunks = self.summary_manager.pending_episode_source
        if not chunks:
            return
        
        # Format chunks as text
        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            events = chunk.get("main_events", [])
            if events:
                chunks_text += f"\nChunk {i}:\n"
                for event in events[:5]:  # Limit to 5 events per chunk
                    if isinstance(event, dict):
                        chunks_text += f"- {event.get('description', event.get('event', ''))}\n"
                    elif isinstance(event, str):
                        chunks_text += f"- {event}\n"
        
        # Generate episodic summary
        prompt = EPISODIC_SUMMARY_PROMPT.format(chunks_text=chunks_text)
        response = self.model.generate(prompt, max_new_tokens=500)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            import json
            import re
            
            # Find JSON block
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                episode_summary = json.loads(json_match.group())
            else:
                # Fallback to empty summary
                episode_summary = {
                    "narrative_arc": "",
                    "major_transitions": [],
                    "emotional_shift": "",
                    "unresolved_elements": []
                }
        except:
            episode_summary = {
                "narrative_arc": "",
                "major_transitions": [],
                "emotional_shift": "",
                "unresolved_elements": []
            }
        
        # Store the episode summary
        self.summary_manager.update_episode_summary(episode_summary)
        self.summary_manager.pending_episode_source = None

    # ------------------------------------------------------------
    # MEMORY UPDATE
    # ------------------------------------------------------------
    def _update_memory(self, new_text, metrics, chunk_id, accumulated_metrics=None):
        """
        Update local, S2 (chunk), E2 (episode), G3 (global) memory.
        And save snapshot for this chunk.
        """
        # NOTE: We still use JSON for internal memory updates, but we don't show this to the generation prompt
        
        # ---- Chunk summary (S2)
        summary_prompt = (
            "Summarize the following text into JSON format.\n"
            "Return ONLY valid JSON with these exact keys:\n"
            "main_events, characters, places, objects, concepts\n\n"
            f"{new_text}\n\n"
            "JSON:"
        )

        summary_raw = self.model.generate(summary_prompt, max_new_tokens=512)
        
        # Better JSON extraction
        try:
            # Try to find JSON block in response
            if "{" in summary_raw and "}" in summary_raw:
                json_start = summary_raw.index("{")
                json_end = summary_raw.rindex("}") + 1
                json_str = summary_raw[json_start:json_end]
                summary = json.loads(json_str)
                
                # Ensure all required keys exist
                for key in ["main_events", "characters", "places", "objects", "concepts"]:
                    if key not in summary:
                        summary[key] = []
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            # Fallback to empty structure
            summary = {
                "main_events": [],
                "characters": [],
                "places": [],
                "objects": [],
                "concepts": [],
            }

        summary["chunk_id"] = chunk_id
        self.summary_manager.add_chunk_summary(summary)
        self.last_chunk_text = str(new_text or "").strip()
        if self.last_chunk_text:
            self.retrieval_text_cache[chunk_id] = self.last_chunk_text

        # ---- Graph memory
        self._update_graph_from_summary(summary)

        # ---- Semantic memory + raw-chunk retrieval
        retrieval_excerpt = self._make_retrieval_excerpt(new_text)
        embedding_text = retrieval_excerpt or new_text[:1000]
        vec = self.model.get_embedding(embedding_text)  # Returns (1, hidden_size)
        vec_np = vec[0].detach().cpu().numpy().astype(float)

        self.summary_manager.add_semantic_vector(vec_np.tolist())
        if retrieval_excerpt:
            self.summary_manager.add_retrieval_chunk({
                "chunk_id": chunk_id,
                "excerpt": retrieval_excerpt,
            })
            self.retrieval_vectors.append(vec_np)

        # ---- Episodic
        if getattr(self.summary_manager, "pending_episode_source", None):
            ep_chunks = self.summary_manager.pending_episode_source

            ep_prompt = (
                "Summarize the following chunk summaries into EPISODIC JSON:\n\n"
                f"{json.dumps(ep_chunks, indent=2)}\n\n"
                "JSON with: narrative_arc, major_transitions, emotional_shift, unresolved_elements\n"
            )

            ep_raw = self.model.generate(ep_prompt, max_new_tokens=512)
            try:
                ep = json.loads(ep_raw)
            except:
                ep = {
                    "narrative_arc": "",
                    "major_transitions": [],
                    "emotional_shift": "",
                    "unresolved_elements": [],
                }

            self.summary_manager.update_episode_summary(ep)
            self.summary_manager.pending_episode_source = None

        # ---- Global summary
        if self.summary_manager.episodes:
            # Only use the current global summary and the LATEST episode to update
            # This prevents the prompt from exploding as memory grows
            current_global = self.summary_manager.global_summary or "No summary yet."
            latest_episode = self.summary_manager.episodes[-1]
            
            g_prompt = (
                "Update the GLOBAL SUMMARY to include the latest episode.\n\n"
                f"CURRENT SUMMARY:\n{current_global}\n\n"
                f"LATEST EPISODE:\n{json.dumps(latest_episode, indent=2)}\n\n"
                "Return ONLY a single updated paragraph:"
            )
            g_raw = self.model.generate(g_prompt, max_new_tokens=512)
            
            # Extract just the summary paragraph
            lines = [l.strip() for l in g_raw.strip().split("\n") if l.strip()]
            filtered = [l for l in lines 
                       if not l.startswith("Update") 
                       and not l.startswith("Return")
                       and not l.startswith("{")
                       and not l.startswith("}")
                       and len(l) > 20]
            
            if filtered:
                g_raw = " ".join(filtered)
            
            self.summary_manager.update_global_summary(g_raw)

        # ---- SAVE MEMORY SNAPSHOT FOR THIS CHUNK
        self.serializer.save_memory_snapshot(
            self.summary_manager, 
            f"chunk_{chunk_id:03d}",
            full_tokens=self.full_tokens,
            accumulated_metrics=accumulated_metrics,
            graph_memory=self.graph_memory.to_serializable(),
        )
