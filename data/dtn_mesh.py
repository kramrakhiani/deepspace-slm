import json
import zlib
import time
from typing import Dict, List, Optional, Any


class InventoryCRDTMesh:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.state: Dict[str, Dict[str, Any]] = {}
        self.vector_clock: Dict[str, int] = {node_id: 0}

    def update_item(self, item_name: str, qty: int, status: str) -> Dict[str, Any]:
        self.vector_clock[self.node_id] = self.vector_clock.get(self.node_id, 0) + 1
        entry = {
            "item_name": item_name.lower(),
            "quantity": qty,
            "status": status,
            "updated_by": self.node_id,
            "version": self.vector_clock[self.node_id],
            "timestamp": time.time(),
        }
        self.state[item_name.lower()] = entry
        return entry

    def merge_remote_state(self, remote_state: Dict[str, Dict[str, Any]]) -> List[str]:
        merged = []
        for item, remote_entry in remote_state.items():
            local_entry = self.state.get(item)
            if not local_entry or remote_entry.get("timestamp", 0) > local_entry.get("timestamp", 0):
                self.state[item] = dict(remote_entry)
                merged.append(item)
        return merged


class DTNBundleQueue:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.queue: List[bytes] = []

    def create_space_packet(self, payload_dict: Dict[str, Any]) -> bytes:
        raw_json = json.dumps(payload_dict).encode("utf-8")
        compressed = zlib.compress(raw_json)

        header = f"CCSDS|SRC:{self.node_id}|LEN:{len(compressed)}|".encode("utf-8")
        packet = header + compressed
        self.queue.append(packet)
        return packet

    def transmit_next(self) -> Optional[Dict[str, Any]]:
        if not self.queue:
            return None
        packet = self.queue.pop(0)
        parts = packet.split(b"|", 3)
        if len(parts) < 4:
            return None
        compressed = parts[3]
        try:
            decompressed = zlib.decompress(compressed)
            return json.loads(decompressed.decode("utf-8"))
        except Exception:
            return None
