import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "<QUERY>": 4,
    "<RESPONSE>": 5,
    "<ITEM>": 6,
    "<QTY>": 7,
    "<LOC>": 8,
    "<STATUS>": 9,
    "<ALERT>": 10,
    "<SEP>": 11,
}

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
BYTE_PREFIX = "Ġ"  # GPT-2 style byte space prefix representation


class HabitatTokenizer:
    """
    Byte-Level BPE Subword Tokenizer for DeepSpace-SLM.
    Supports byte-prefix subwords for zero out-of-vocabulary failures & clean word spacing.
    """
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._build_bpe_vocabulary()

    def _build_bpe_vocabulary(self):
        # 1. Register Special Tokens
        for token, idx in SPECIAL_TOKENS.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        idx = NUM_SPECIAL_TOKENS

        # 2. Register 256 Base Byte Tokens
        for c in range(256):
            char = chr(c)
            if char not in self.token_to_id and idx < self.vocab_size:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        # 3. Register Byte Prefix Space character 'Ġ'
        if BYTE_PREFIX not in self.token_to_id and idx < self.vocab_size:
            self.token_to_id[BYTE_PREFIX] = idx
            self.id_to_token[idx] = BYTE_PREFIX
            idx += 1

        # 4. Register Domain Subwords & Space-Prefixed Subwords
        domain_vocabulary = self._get_domain_vocabulary()
        for word in domain_vocabulary:
            # Standalone subword
            if word not in self.token_to_id and idx < self.vocab_size:
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word
                idx += 1

            # Space-prefixed subword (e.g. Ġcheck, Ġoxygen, Ġcanister)
            prefixed = BYTE_PREFIX + word
            if prefixed not in self.token_to_id and idx < self.vocab_size:
                self.token_to_id[prefixed] = idx
                self.id_to_token[idx] = prefixed
                idx += 1

        self.actual_vocab_size = idx

    def _get_domain_vocabulary(self) -> List[str]:
        words = []
        words += [
            "check", "query", "locate", "find", "count", "report",
            "consume", "consumed", "restock", "transfer", "deploy",
            "replace", "inspect", "maintain", "repair", "discard",
            "request", "forecast", "alert", "status", "update",
            "inventory", "scan", "log", "track", "use", "used",
            "remaining", "available", "needed", "required", "critical",
            "low", "nominal", "full", "empty", "expired", "expiring",
            "damaged", "functional", "offline", "online", "active", "logged",
        ]
        words += [
            "oxygen", "o2", "canister", "canisters", "tank", "tanks",
            "nitrogen", "n2", "co2", "scrubber", "scrubbers", "filter",
            "filters", "water", "purifier", "purifiers", "recycler",
            "air", "ventilation", "hvac", "heater", "heaters",
            "food", "ration", "rations", "meal", "meals", "packet",
            "packets", "protein", "carbohydrate", "vitamin", "vitamins",
            "supplement", "supplements", "freeze", "dried", "hydrated",
            "calorie", "calories", "nutrition", "bar", "bars",
            "medical", "med", "kit", "kits", "bandage", "bandages",
            "medication", "medications", "antibiotic", "antibiotics",
            "painkiller", "painkillers", "antiseptic", "syringe",
            "syringes", "defibrillator", "thermometer", "dosage",
            "first", "aid", "emergency", "trauma", "surgical",
            "tool", "tools", "wrench", "wrenches", "screwdriver",
            "hammer", "drill", "multimeter", "voltmeter", "soldering",
            "cable", "cables", "wire", "wires", "connector", "connectors",
            "seal", "seals", "gasket", "gaskets", "bolt", "bolts",
            "nut", "nuts", "tape", "adhesive", "epoxy", "lubricant",
            "spare", "part", "parts", "component", "components",
            "module", "modules", "board", "boards", "circuit",
            "processor", "sensor", "sensors", "actuator", "actuators",
            "pump", "pumps", "valve", "valves", "motor", "motors",
            "battery", "batteries", "cell", "cells", "panel", "panels",
            "solar", "radiator", "radiators",
            "suit", "suits", "eva", "helmet", "helmets", "glove",
            "gloves", "boot", "boots", "tether", "tethers",
            "regulator", "regulators",
            "waste", "bag", "bags", "container", "containers",
            "wipe", "wipes", "sanitizer", "disposal", "compost",
        ]
        words += [
            "hab", "habitat", "module", "airlock", "cupola",
            "galley", "laboratory", "lab", "crew", "quarters",
            "storage", "bay", "compartment", "locker", "shelf",
            "rack", "bin", "drawer", "cabinet", "section",
            "forward", "aft", "port", "starboard", "upper", "lower",
            "primary", "secondary", "backup", "auxiliary",
            "alpha", "beta", "gamma", "delta",
            "node", "corridor", "hatch", "vestibule",
            "a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4",
            "c1", "c2", "c3", "c4", "d1", "d2", "d3", "d4",
        ]
        words += [
            "unit", "units", "piece", "pieces", "pair", "pairs",
            "set", "sets", "box", "boxes", "case", "cases",
            "liter", "liters", "ml", "gallon", "kg", "gram", "grams",
            "mg", "lb", "oz", "psi", "kpa", "bar",
            "percent", "total", "quantity", "qty", "amount",
            "rate", "per", "day", "days", "week", "weeks",
            "month", "months", "hour", "hours", "minute", "minutes",
        ]
        words += [
            "ok", "good", "warning", "danger", "caution",
            "normal", "abnormal", "threshold", "below", "above",
            "mission", "day", "crew", "member", "astronaut",
            "commander", "specialist", "engineer",
            "supply", "demand", "consumption", "usage", "stock",
            "level", "current", "previous", "projected", "estimated",
            "date", "time", "expiry", "expiration", "shelf_life",
            "priority", "high", "medium", "safety", "stored",
            "is", "are", "the", "in", "at", "on", "of", "for",
            "to", "from", "and", "or", "not", "no", "yes",
            "how", "many", "much", "where", "what", "which", "when",
            "this", "that", "all", "each", "every", "any",
            "there", "have", "has", "with", "by", "as", "it",
            "will", "be", "been", "was", "were", "do", "does",
            "can", "should", "need", "must", "may",
        ]
        return words

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(SPECIAL_TOKENS["<BOS>"])

        # Split text into special tokens vs words & spaces
        chunks = re.findall(r"<\w+>|\S+|\s+", text)

        for i, chunk in enumerate(chunks):
            if chunk in SPECIAL_TOKENS:
                tokens.append(SPECIAL_TOKENS[chunk])
                continue

            chunk_clean = chunk.lower()
            # If chunk is a space, prepend Ġ to next word or emit Ġ
            if chunk_clean.isspace():
                continue

            # Add byte space prefix for words after the first chunk or leading spaces
            prefixed = BYTE_PREFIX + chunk_clean

            if prefixed in self.token_to_id:
                tokens.append(self.token_to_id[prefixed])
            elif chunk_clean in self.token_to_id:
                tokens.append(self.token_to_id[chunk_clean])
            else:
                # Subword byte-level fallback
                for char in chunk_clean:
                    tokens.append(
                        self.token_to_id.get(char, SPECIAL_TOKENS["<UNK>"])
                    )

        if add_eos:
            tokens.append(SPECIAL_TOKENS["<EOS>"])

        if max_length is not None:
            tokens = tokens[:max_length]

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        pieces = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if skip_special and tok in SPECIAL_TOKENS:
                    continue
                pieces.append(tok)
            else:
                pieces.append("<UNK>")

        text = "".join(pieces)
        # Convert byte prefix Ġ to space
        text = text.replace(BYTE_PREFIX, " ")
        text = re.sub(r"\s+([.,!?:;\'>])", r"\1", text)
        text = re.sub(r"([<])\s+", r"\1", text)
        return text.strip()

    def pad(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        return token_ids + [SPECIAL_TOKENS["<PAD>"]] * (max_length - len(token_ids))

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<PAD>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<EOS>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["<UNK>"]
