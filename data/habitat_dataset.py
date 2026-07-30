import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from data.tokenizer import HabitatTokenizer

INVENTORY_ITEMS = [
    ("o2 canister", "life_support", "units", 48, "critical"),
    ("n2 tank", "life_support", "units", 12, "critical"),
    ("co2 scrubber", "life_support", "units", 24, "critical"),
    ("air filter", "life_support", "units", 36, "high"),
    ("water filter", "life_support", "units", 20, "critical"),
    ("water purifier cartridge", "life_support", "units", 8, "critical"),
    ("food ration", "nutrition", "packets", 720, "critical"),
    ("protein bar", "nutrition", "bars", 200, "medium"),
    ("vitamin supplement", "nutrition", "bottles", 30, "medium"),
    ("freeze dried meal", "nutrition", "packets", 480, "high"),
    ("medical kit", "medical", "kits", 6, "critical"),
    ("bandage roll", "medical", "units", 50, "high"),
    ("antibiotic pack", "medical", "packs", 24, "critical"),
    ("painkiller bottle", "medical", "bottles", 12, "high"),
    ("antiseptic wipe", "medical", "boxes", 30, "medium"),
    ("syringe", "medical", "units", 40, "high"),
    ("wrench set", "tools", "sets", 4, "medium"),
    ("multimeter", "tools", "units", 3, "medium"),
    ("soldering kit", "tools", "kits", 2, "medium"),
    ("cable bundle", "tools", "bundles", 15, "low"),
    ("seal gasket", "spare_parts", "units", 60, "high"),
    ("circuit board", "spare_parts", "units", 10, "high"),
    ("pump assembly", "spare_parts", "units", 4, "critical"),
    ("valve actuator", "spare_parts", "units", 8, "high"),
    ("battery cell", "spare_parts", "cells", 32, "high"),
    ("solar panel module", "spare_parts", "units", 4, "medium"),
    ("eva suit patch kit", "eva", "kits", 6, "critical"),
    ("suit helmet visor", "eva", "units", 3, "high"),
    ("tether cable", "eva", "units", 8, "high"),
    ("waste disposal bag", "waste", "boxes", 100, "medium"),
    ("sanitizer bottle", "waste", "bottles", 20, "low"),
]

LOCATIONS = [
    "storage bay a1", "storage bay a2", "storage bay b1", "storage bay b2",
    "module alpha locker 1", "module alpha locker 2",
    "module beta locker 1", "module beta locker 2",
    "module gamma shelf 1", "module gamma shelf 2",
    "crew quarters cabinet 1", "crew quarters cabinet 2",
    "laboratory rack 1", "laboratory rack 2",
    "airlock compartment 1", "galley storage 1",
    "medical bay cabinet 1", "medical bay cabinet 2",
    "forward node bin 1", "aft node bin 1",
]

# Comprehensive query phrasing templates for natural language generalization
STOCK_QUERY_TEMPLATES = [
    "how much {item} left",
    "how many {item} do we have",
    "what is the stock of {item}",
    "check stock on {item}",
    "do we have any {item} remaining",
    "how many {item} are available",
    "what is our current count of {item}",
    "report quantity of {item}",
    "how much {item} is remaining in habitat",
    "is there any {item} left",
    "give me the current count of {item}",
    "check inventory level of {item}",
    "what is the supply level of {item}",
    "do we have enough {item}",
    "how much {item} is stored in habitat",
]

LOCATE_QUERY_TEMPLATES = [
    "where is {item} stored",
    "where can I find {item}",
    "locate {item}",
    "where is the {item} kept",
    "which locker holds {item}",
    "find location of {item}",
    "where can crew find {item}",
    "which compartment has {item}",
    "where are {item} located",
    "tell me where {item} is",
]

FORECAST_QUERY_TEMPLATES = [
    "how long will {item} last",
    "when will we run out of {item}",
    "forecast {item} consumption",
    "what is the usage rate of {item}",
    "how many days remaining for {item}",
    "how fast are we consuming {item}",
    "estimated duration for {item} supply",
    "when do we need to restock {item}",
]

PROCEDURE_QUERY_TEMPLATES = [
    "how to replace {item}",
    "procedure to inspect {item}",
    "how to calibrate {item}",
    "maintenance steps for {item}",
    "how to clean {item}",
    "recalibration manual for {item}",
]

RESPONSE_TEMPLATES = [
    "we currently have {qty} {unit} of {item} stored in {loc}. status is {status}.",
    "{item} count is {qty} {unit} located in {loc}. system status is {status}.",
    "inventory report for {item}: {qty} {unit} available in {loc}. status is {status}.",
    "there are {qty} {unit} of {item} remaining in {loc}. status: {status}.",
]


def _random_qty(typical: int, rng: Optional[random.Random] = None) -> int:
    r = rng or random
    return max(0, int(r.gauss(typical * 0.5, typical * 0.25)))


def _random_days_remaining(qty: int, rate: float) -> int:
    if rate <= 0:
        return 999
    return max(0, int(qty / rate))


class InventoryState:
    def __init__(self, seed: Optional[int] = None):
        rng = random.Random(seed)
        self.items: Dict[str, Dict] = {}
        self.mission_day = rng.randint(30, 500)

        for name, category, unit, typical_qty, criticality in INVENTORY_ITEMS:
            qty = _random_qty(typical_qty, rng)
            location = rng.choice(LOCATIONS)
            consumption_rate = max(0.1, rng.gauss(typical_qty / 180, typical_qty / 360))

            self.items[name] = {
                "name": name,
                "category": category,
                "unit": unit,
                "quantity": qty,
                "max_quantity": typical_qty,
                "location": location,
                "criticality": criticality,
                "consumption_rate": round(consumption_rate, 2),
                "days_remaining": _random_days_remaining(qty, consumption_rate),
                "status": self._status(qty, typical_qty),
            }

    def _status(self, qty: int, max_qty: int) -> str:
        ratio = qty / max(max_qty, 1)
        if ratio <= 0:
            return "empty"
        elif ratio <= 0.1:
            return "critical"
        elif ratio <= 0.25:
            return "low"
        elif ratio <= 0.75:
            return "nominal"
        else:
            return "full"


class HabitatInventoryDataset(Dataset):
    def __init__(
        self,
        tokenizer: HabitatTokenizer,
        num_samples: int = 5000,
        max_seq_len: int = 64,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        self.examples = self._generate_examples()

    def _generate_examples(self) -> List[Tuple[str, str]]:
        examples = []
        for i in range(self.num_samples):
            state = InventoryState(seed=self.rng.randint(0, 1000000))
            item_name, item_info = self.rng.choice(list(state.items.items()))

            template_type = self.rng.choice(["stock", "locate", "forecast", "alert", "log"])

            if template_type == "stock":
                q = f"<QUERY> check stock {item_name}"
                r = f"<RESPONSE> <ITEM> {item_info['name']} <QTY> {item_info['quantity']} {item_info['unit']} <STATUS> {item_info['status']} <LOC> {item_info['location']}"

            elif template_type == "locate":
                q = f"<QUERY> locate {item_name}"
                r = f"<RESPONSE> <ITEM> {item_info['name']} <LOC> {item_info['location']} <QTY> {item_info['quantity']} {item_info['unit']} available"

            elif template_type == "forecast":
                q = f"<QUERY> forecast {item_name} consumption"
                r = f"<RESPONSE> <ITEM> {item_info['name']} at rate of {item_info['consumption_rate']} {item_info['unit']}/day, estimated {item_info['days_remaining']} days remaining"

            elif template_type == "alert":
                low_items = [v for v in state.items.values() if v["status"] in ("critical", "low", "empty")]
                if low_items:
                    chosen = self.rng.sample(low_items, min(len(low_items), 3))
                    alerts_str = " <SEP> ".join([f"<ALERT> {it['name']}: {it['quantity']} {it['unit']} ({it['status']})" for it in chosen])
                    q = "<QUERY> what items are critically low"
                    r = f"<RESPONSE> {alerts_str}"
                else:
                    q = "<QUERY> what items are critically low"
                    r = "<RESPONSE> all inventory levels are nominal. no alerts"

            else:
                act = self.rng.choice(["consumed", "restocked", "inspected", "repaired"])
                q = f"<QUERY> log {act} {item_name}"
                r = f"<RESPONSE> logged: {item_name} {act} at {item_info['location']} on mission day {state.mission_day}"

            examples.append((q, r))

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query, response = self.examples[idx]
        full_text = f"{query} {response}"

        token_ids = self.tokenizer.encode(
            full_text,
            add_bos=True,
            add_eos=True,
            max_length=self.max_seq_len,
        )

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        pad_id = self.tokenizer.pad_id
        input_ids = self.tokenizer.pad(input_ids, self.max_seq_len)
        target_ids = self.tokenizer.pad(target_ids, self.max_seq_len)

        attention_mask = [1 if tid != pad_id else 0 for tid in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
        }


class HabitatConversationalDataset(Dataset):
    """
    Multi-Turn Conversational Space Habitat Dialogue Dataset with Rich Paraphrasing & Phrasing Diversity.
    Generates 20,000+ conversational turns across all 31 habitat items and diverse user phrasings.
    """
    def __init__(
        self,
        tokenizer: HabitatTokenizer,
        num_samples: int = 20000,
        max_seq_len: int = 64,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        self.examples = self._generate_conversational_turns()

    def _generate_conversational_turns(self) -> List[Tuple[str, str]]:
        examples = []
        for i in range(self.num_samples):
            state = InventoryState(seed=self.rng.randint(0, 1000000))
            item_name, item_info = self.rng.choice(list(state.items.items()))

            turn_type = self.rng.choice(["prose_stock", "prose_locate", "prose_forecast", "prose_procedure", "prose_alert"])

            if turn_type == "prose_stock":
                template = self.rng.choice(STOCK_QUERY_TEMPLATES)
                q_text = template.format(item=item_name)

                resp_tpl = self.rng.choice(RESPONSE_TEMPLATES)
                r_text = resp_tpl.format(
                    item=item_info['name'],
                    qty=item_info['quantity'],
                    unit=item_info['unit'],
                    loc=item_info['location'],
                    status=item_info['status'],
                )

            elif turn_type == "prose_locate":
                template = self.rng.choice(LOCATE_QUERY_TEMPLATES)
                q_text = template.format(item=item_name)
                r_text = f"{item_info['name']} is stored in {item_info['location']} with {item_info['quantity']} {item_info['unit']} available."

            elif turn_type == "prose_forecast":
                template = self.rng.choice(FORECAST_QUERY_TEMPLATES)
                q_text = template.format(item=item_name)
                r_text = f"at consumption rate of {item_info['consumption_rate']} {item_info['unit']} per day, {item_info['name']} is estimated to last {item_info['days_remaining']} days."

            elif turn_type == "prose_procedure":
                template = self.rng.choice(PROCEDURE_QUERY_TEMPLATES)
                q_text = template.format(item=item_name)
                r_text = f"procedure for {item_info['name']}: unlatch retention lock in {item_info['location']}, replace unit, verify nominal status."

            else:
                q_text = "check life support alerts"
                r_text = "all habitat supply levels are nominal. no active alerts."

            q = f"<QUERY> {q_text}"
            r = f"<RESPONSE> {r_text}"

            examples.append((q, r))

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query, response = self.examples[idx]
        full_text = f"{query} {response}"

        token_ids = self.tokenizer.encode(
            full_text,
            add_bos=True,
            add_eos=True,
            max_length=self.max_seq_len,
        )

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        # Query loss masking: mask out prompt tokens so loss is computed ONLY on response
        query_token_ids = self.tokenizer.encode(query, add_bos=True, add_eos=False)
        query_len = len(query_token_ids)

        # Set prompt target tokens to -100 (ignored by loss)
        target_ids_masked = list(target_ids)
        for k in range(min(query_len - 1, len(target_ids_masked))):
            target_ids_masked[k] = -100

        pad_id = self.tokenizer.pad_id
        input_ids = self.tokenizer.pad(input_ids, self.max_seq_len)

        # Pad masked targets with -100
        if len(target_ids_masked) < self.max_seq_len:
            target_ids_masked += [-100] * (self.max_seq_len - len(target_ids_masked))
        else:
            target_ids_masked = target_ids_masked[:self.max_seq_len]

        attention_mask = [1 if tid != pad_id else 0 for tid in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids_masked, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
        }


def create_dataloader(
    tokenizer: HabitatTokenizer,
    num_samples: int = 5000,
    max_seq_len: int = 64,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    conversational: bool = False,
) -> DataLoader:
    if conversational:
        dataset = HabitatConversationalDataset(
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            seed=seed,
        )
    else:
        dataset = HabitatInventoryDataset(
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            seed=seed,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
