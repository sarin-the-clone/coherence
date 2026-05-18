import json
from typing import Dict, List, Any
import numpy as np

class MemorySummarizer:
    """Converts structured memory JSON into natural language summaries."""

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    @staticmethod
    def _find_similar_chunks(current_vector, semantic_memory, recent_chunks, top_k=2):
        """Find top-k most semantically similar past chunks."""
        if not semantic_memory or not semantic_memory.get("vectors"):
            return []
        
        vectors = semantic_memory["vectors"]
        n_chunks = len(recent_chunks)
        
        # Don't compare with the very last chunk (it's the current one)
        if len(vectors) > n_chunks:
            vectors = vectors[:n_chunks-1]
        
        # Compute similarities
        similarities = []
        for i, vec in enumerate(vectors):
            sim = MemorySummarizer._cosine_similarity(current_vector, vec)
            similarities.append((i, sim))
        
        # Sort by similarity (descending) and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, sim in similarities[:top_k] if sim > 0.3]  # threshold
        
        return top_indices

    @staticmethod
    def _clip_excerpt(text, max_chars=420):
        text = " ".join(str(text).split())
        if len(text) <= max_chars:
            return text

        clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
        if not clipped:
            clipped = text[:max_chars].strip()
        return clipped + " ..."

    @staticmethod
    def _summarize_graph(graph_data, max_nodes=8, max_edges=5):
        if not isinstance(graph_data, dict):
            return []

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        if not nodes:
            return []

        node_lookup = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            try:
                node_id = int(node.get("id"))
            except (TypeError, ValueError):
                continue
            node_lookup[node_id] = node

        sorted_nodes = sorted(
            [n for n in node_lookup.values() if str(n.get("label", "")).strip()],
            key=lambda n: (int(n.get("recency", 0)), -int(n.get("frequency", 1)), str(n.get("label", "")).lower())
        )[:max_nodes]

        grouped = {}
        for node in sorted_nodes:
            node_type = str(node.get("type", "concept"))
            grouped.setdefault(node_type, [])
            grouped[node_type].append(str(node.get("label", "")).strip())

        lines = ["**Entity Graph:**"]
        display_order = ["character", "place", "object", "concept", "event"]
        display_names = {
            "character": "Characters",
            "place": "Places",
            "object": "Objects",
            "concept": "Concepts",
            "event": "Event Threads",
        }
        for key in display_order:
            labels = grouped.get(key)
            if labels:
                lines.append(f"- {display_names[key]}: {', '.join(labels[:4])}")

        sorted_edges = sorted(
            [e for e in edges if isinstance(e, dict)],
            key=lambda e: (int(e.get("recency", 0)), -int(e.get("frequency", 1)))
        )

        edge_lines = []
        for edge in sorted_edges:
            src = node_lookup.get(edge.get("source"))
            dst = node_lookup.get(edge.get("target"))
            if not src or not dst:
                continue

            src_label = str(src.get("label", "")).strip()
            dst_label = str(dst.get("label", "")).strip()
            if not src_label or not dst_label:
                continue

            relation = str(edge.get("relation", "related_to")).replace("_", " ").strip()
            edge_lines.append(f"- {src_label} {relation} {dst_label}")
            if len(edge_lines) >= max_edges:
                break

        if edge_lines:
            lines.append("**Key Relationships:**")
            lines.extend(edge_lines)
        lines.append("")
        return lines

    @staticmethod
    def _render_retrieved_chunks(retrieved_chunks, max_items=2):
        if not isinstance(retrieved_chunks, list) or not retrieved_chunks:
            return []

        lines = ["**Retrieved Earlier Passages:**"]
        added = 0
        for rec in retrieved_chunks:
            if not isinstance(rec, dict):
                continue
            excerpt = str(rec.get("prompt_excerpt") or rec.get("excerpt", "")).strip()
            if not excerpt:
                continue

            try:
                chunk_id = int(rec.get("chunk_id"))
                label = f"Chunk {chunk_id}"
            except (TypeError, ValueError):
                label = "Earlier passage"

            lines.append(f"{label}: {excerpt}")
            added += 1
            if added >= max_items:
                break

        if added == 0:
            return []

        lines.append("")
        return lines

    @staticmethod
    def summarize_for_generation(memory: Dict[str, Any], current_semantic_vector=None) -> str:
        """Creates a comprehensive summary for story generation with rich narrative context."""
        summary_parts = []
        
        # 1. Overall Story Arc (from global summary or episodes)
        global_summary = memory.get("global_summary", "")
        if isinstance(global_summary, str) and global_summary.strip():
            if not (global_summary.strip().startswith('"') or global_summary.strip().startswith('{')):
                summary_parts.append(f"**Story So Far:**\n{global_summary}\n")
        
        # 2. Recent Plot Progression (last 3 chunks in narrative form)
        recent_chunks = memory.get("recent_chunks", [])
        if recent_chunks:
            summary_parts.append("**Recent Plot Developments:**")
            # Take last 3 chunks to give model more context
            visible_chunks = recent_chunks[-3:]
            fallback_start = max(1, len(recent_chunks) - len(visible_chunks) + 1)
            for i, chunk in enumerate(visible_chunks):
                try:
                    chunk_num = int(chunk.get("chunk_id"))
                except (AttributeError, TypeError, ValueError):
                    chunk_num = fallback_start + i
                events = chunk.get("main_events", [])
                if events:
                    event_text = []
                    # Handle if events is a dict (unlikely but possible from LLM)
                    if isinstance(events, dict):
                        events = [events]
                    elif not isinstance(events, list):
                        continue

                    for event in events:
                        if isinstance(event, dict):
                            # Extract description, ensuring it's a string
                            desc = event.get("details", event.get("description", event.get("event", "")))
                            if isinstance(desc, list):
                                desc = " ".join(str(d) for d in desc)
                            elif not isinstance(desc, str):
                                desc = str(desc)
                            
                            if desc:
                                event_text.append(desc)
                        elif isinstance(event, str):
                            event_text.append(event)
                        elif isinstance(event, list):
                            event_text.append(" ".join(str(e) for e in event))
                    
                    if event_text:
                        summary_parts.append(f"Chunk {chunk_num}: {' '.join(event_text)}")
            summary_parts.append("")  # blank line

        # 3. Raw-text long-range retrieval
        summary_parts.extend(
            MemorySummarizer._render_retrieved_chunks(memory.get("retrieved_chunks", []))
        )

        # 4. Entity graph
        summary_parts.extend(
            MemorySummarizer._summarize_graph(memory.get("graph_memory", {}))
        )

        # 5. Active Characters with Context
        characters = []
        char_details = {}
        # Collect characters from last 2 chunks to get more context
        for chunk in recent_chunks[-2:]:
            chars = chunk.get("characters", [])
            for char in chars:
                if isinstance(char, dict):
                    name = char.get("name", "")
                    role = char.get("role", "")
                    if name:
                        char_details[name] = role
                elif isinstance(char, str):
                    char_details[char] = ""
        
        if char_details:
            summary_parts.append("**Active Characters:**")
            for name, role in char_details.items():
                if role:
                    summary_parts.append(f"- {name}: {role}")
                else:
                    summary_parts.append(f"- {name}")
            summary_parts.append("")

        # 6. Current Setting & Atmosphere
        places = []
        for chunk in recent_chunks[-2:]:
            locs = chunk.get("places", [])
            for loc in locs:
                if isinstance(loc, dict):
                    name = loc.get("name", "")
                    if name and name not in places:
                        places.append(name)
                elif isinstance(loc, str) and loc not in places:
                    places.append(loc)
        
        if places:
            summary_parts.append(f"**Current Setting:** {', '.join(places)}\n")
        
        # 7. Narrative Momentum (from episodic summaries if available)
        episodes = memory.get("episodes", [])
        if episodes:
            last_episode = episodes[-1]
            # Handle if last_episode is a list (rare but possible from LLM JSON)
            if isinstance(last_episode, list):
                if last_episode and isinstance(last_episode[0], dict):
                    last_episode = last_episode[0]
                else:
                    last_episode = {}
            
            if isinstance(last_episode, dict):
                narrative_arc = last_episode.get("narrative_arc", "")
                unresolved = last_episode.get("unresolved_elements", [])
            else:
                narrative_arc = ""
                unresolved = []
            
            if narrative_arc:
                summary_parts.append(f"**Narrative Arc:** {narrative_arc}\n")
            
            if unresolved:
                summary_parts.append("**Unresolved Threads:**")
                for item in unresolved:
                    if isinstance(item, str):
                        summary_parts.append(f"- {item}")
                summary_parts.append("")

        return "\n".join(summary_parts)

    @staticmethod
    def summarize_for_title(memory: Dict[str, Any]) -> str:
        """
        Creates a focused summary for title/genre extraction.
        CRITICAL FIX: Ignores recent chunks to avoid recency bias.
        Uses Global Summary + Original Intent + Narrative Arc.
        """
        summary_parts = []

        # 1. Original Intent (The Prompt)
        initial_prompt = memory.get("initial_prompt", "")
        prompt_filename = memory.get("prompt_filename", "")
        
        if initial_prompt:
             # Truncate if insanely long, but usually prompts are key
             summary_parts.append(f"**ORIGINAL BOOK INTENT/PROMPT:**\n{initial_prompt[:2000]}\n")
        
        if prompt_filename:
             summary_parts.append(f"**Source Category/Filename:** {prompt_filename}\n")

        # 2. Global Story Summary (The "Book Jacket" summary)
        global_summary = memory.get("global_summary", "")
        if isinstance(global_summary, str) and global_summary.strip():
            # Clean up artifacts
            clean_summary = global_summary.strip()
            if not (clean_summary.startswith('{') or clean_summary.startswith('"')):
                 summary_parts.append(f"**Story Overview:**\n{clean_summary}\n")

        # 3. Full Narrative Arc (from Episodes)
        # We want the trajectory, not just the last scene
        episodes = memory.get("episodes", [])
        if episodes:
            summary_parts.append("**Narrative Arc / Evolution:**")
            # Take first, middle, and last episode arcs if available
            indices = []
            if len(episodes) > 0: indices.append(0)
            if len(episodes) > 2: indices.append(len(episodes) // 2)
            if len(episodes) > 1: indices.append(len(episodes) - 1)
            
            # Deduplicate indices
            indices = sorted(list(set(indices)))
            
            for i in indices:
                ep = episodes[i]
                if isinstance(ep, list) and ep: ep = ep[0] # handle list wrapper
                if isinstance(ep, dict):
                     arc = ep.get("narrative_arc", "")
                     if arc:
                         label = "Beginning" if i == 0 else ("Ending" if i == len(episodes)-1 else "Middle")
                         summary_parts.append(f"- {label}: {arc}")

        return "\n".join(summary_parts)
