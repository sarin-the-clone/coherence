# ================================================================
# graph_memory.py (v3)
#
# Memory graph for long-form text generation:
#   - Nodes: characters, places, objects, concepts
#   - Edges: relations between nodes (typed)
#   - Recency/frequency tracking
#   - Auto-merge for repeated names
#   - J3 pruning for compact JSON
#
# The longbook engine updates this graph after every chunk.
# ================================================================

import re
import json
from collections import defaultdict


class GraphMemory:
    """
    Graph of entities and relations extracted from narrative text.

    Nodes have:
      - id
      - label
      - type (character, place, object, concept)
      - frequency score
      - recency counter

    Edges have:
      - source id
      - target id
      - relation type ("meets", "travels_to", "fights", ...)
      - frequency
      - recency
    """

    def __init__(self, max_nodes=40, max_edges=80):
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.nodes = {}        # id → node dict
        self.edges = {}        # id → edge dict
        self.next_node_id = 1
        self.next_edge_id = 1

    # ---------------------------------------------------------------
    # Node creation / lookup
    # ---------------------------------------------------------------
    def _find_existing_node(self, label):
        """
        Try to find an existing node with similar label (case-insensitive),
        to avoid duplication of characters/places/etc.
        """
        label_norm = label.lower().strip()
        for node_id, nd in self.nodes.items():
            if nd["label"].lower() == label_norm:
                return node_id
        return None

    def _create_node(self, label, node_type="concept"):
        node_id = self.next_node_id
        self.next_node_id += 1

        self.nodes[node_id] = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "frequency": 1,
            "recency": 0,   # reset by update calls
        }
        return node_id

    # ---------------------------------------------------------------
    # Public node interface
    # ---------------------------------------------------------------
    def register_node(self, label, node_type="concept"):
        """
        Used by longbook_engine to add a character/place/object/concept.
        Always merges with existing nodes if label matches.
        """
        label = label.strip()
        if not label:
            return None

        existing = self._find_existing_node(label)
        if existing is not None:
            self.nodes[existing]["frequency"] += 1
            self.nodes[existing]["recency"] = 0
            return existing

        # Create new node
        new_id = self._create_node(label, node_type=node_type)
        return new_id

    # ---------------------------------------------------------------
    # Edge creation / lookup
    # ---------------------------------------------------------------
    def register_edge(self, src_id, dst_id, relation):
        """
        Register or merge an edge between two nodes.
        """
        if src_id is None or dst_id is None:
            return None
        key = (src_id, dst_id, relation.lower().strip())

        # Check if edge exists
        for edge_id, ed in self.edges.items():
            if (ed["source"], ed["target"], ed["relation"]) == key:
                ed["frequency"] += 1
                ed["recency"] = 0
                return edge_id

        # Create new edge
        edge_id = self.next_edge_id
        self.next_edge_id += 1
        self.edges[edge_id] = {
            "id": edge_id,
            "source": src_id,
            "target": dst_id,
            "relation": relation.lower().strip(),
            "frequency": 1,
            "recency": 0
        }
        return edge_id

    # ---------------------------------------------------------------
    # Aging
    # ---------------------------------------------------------------
    def age_all(self):
        """Increase recency counters for all nodes/edges."""
        for nd in self.nodes.values():
            nd["recency"] += 1
        for ed in self.edges.values():
            ed["recency"] += 1

    # ---------------------------------------------------------------
    # J3 pruning
    # ---------------------------------------------------------------
    def prune(self):
        """
        Keep the top-K most relevant nodes/edges:
          - lowest recency (most recent)
          - highest frequency
        """
        # Nodes first
        if len(self.nodes) > self.max_nodes:
            nodes_sorted = sorted(
                self.nodes.values(),
                key=lambda n: (n["recency"], -n["frequency"])
            )
            keep_nodes = nodes_sorted[:self.max_nodes]
            keep_ids = {nd["id"] for nd in keep_nodes}
            self.nodes = {nid: self.nodes[nid] for nid in keep_ids}
        else:
            keep_ids = set(self.nodes.keys())

        # Edges next — only keep edges that connect kept nodes
        valid_edges = [
            ed for ed in self.edges.values()
            if ed["source"] in keep_ids and ed["target"] in keep_ids
        ]
        valid_edges_sorted = sorted(
            valid_edges,
            key=lambda e: (e["recency"], -e["frequency"])
        )

        if len(valid_edges_sorted) > self.max_edges:
            valid_edges_sorted = valid_edges_sorted[:self.max_edges]

        # rebuild edge dict
        self.edges = {ed["id"]: ed for ed in valid_edges_sorted}

    # ---------------------------------------------------------------
    # JSON serialization
    # ---------------------------------------------------------------
    def to_serializable(self):
        """Return JSON-friendly structure."""
        return {
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges.values())
        }

    # ---------------------------------------------------------------
    def restore_from_serializable(self, data):
        """Restore graph state from serialized JSON-friendly data."""
        if not isinstance(data, dict):
            return

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        self.nodes = {}
        self.edges = {}

        max_node_id = 0
        for node in nodes:
            if not isinstance(node, dict):
                continue
            try:
                node_id = int(node.get("id"))
            except (TypeError, ValueError):
                continue

            restored = {
                "id": node_id,
                "label": str(node.get("label", "")).strip(),
                "type": str(node.get("type", "concept")).strip() or "concept",
                "frequency": int(node.get("frequency", 1)),
                "recency": int(node.get("recency", 0)),
            }
            self.nodes[node_id] = restored
            max_node_id = max(max_node_id, node_id)

        max_edge_id = 0
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            try:
                edge_id = int(edge.get("id"))
                source = int(edge.get("source"))
                target = int(edge.get("target"))
            except (TypeError, ValueError):
                continue

            restored = {
                "id": edge_id,
                "source": source,
                "target": target,
                "relation": str(edge.get("relation", "related_to")).strip() or "related_to",
                "frequency": int(edge.get("frequency", 1)),
                "recency": int(edge.get("recency", 0)),
            }
            self.edges[edge_id] = restored
            max_edge_id = max(max_edge_id, edge_id)

        self.next_node_id = max_node_id + 1
        self.next_edge_id = max_edge_id + 1

    # ---------------------------------------------------------------
    def dump_to_folder(self, folder_path, prefix="memory_graph"):
        """Save a JSON dump for ED64 debugging."""
        import os
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, f"{prefix}.json"), "w") as f:
            json.dump(self.to_serializable(), f, indent=2)
