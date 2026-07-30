import re
from typing import List, Dict, Optional


HABITAT_MANUAL_ENTRIES = [
    {
        "title": "O2 Canister Replacement Procedure",
        "topic": "o2 canister",
        "content": "Step 1: Isolate primary valve. Step 2: Unlatch retention lock in Storage Bay A1. Step 3: Replace empty canister with fresh unit. Step 4: Verify pressure above 2000 PSI."
    },
    {
        "title": "CO2 Scrubber Recalibration",
        "topic": "co2 scrubber",
        "content": "Step 1: Check delta-P sensor reading. Step 2: Flush scrubbers with N2 sweep gas. Step 3: Verify absorption efficiency exceeds 98%."
    },
    {
        "title": "EVA Suit Patch Kit Deployment",
        "topic": "eva suit patch kit",
        "content": "Step 1: Clean suit tear area with isopropyl wipe. Step 2: Apply thermal adhesive patch. Step 3: Cure under UV lamp for 120 seconds. Step 4: Leak test at 4.3 PSI differential."
    },
    {
        "title": "Water Filter Cartridge Flush",
        "topic": "water filter",
        "content": "Step 1: Shut off galley feed line. Step 2: Remove spent cartridge from Storage Bay B1. Step 3: Install primary backup. Step 4: Cycle 5 liters of flush water through waste collector."
    }
]


class HabitatManualRetriever:
    def __init__(self, entries: Optional[List[Dict[str, str]]] = None):
        self.entries = entries or HABITAT_MANUAL_ENTRIES

    def search(self, query: str) -> List[Dict[str, str]]:
        query_words = set(re.findall(r"\w+", query.lower()))
        results = []
        for entry in self.entries:
            text = (entry["title"] + " " + entry["topic"] + " " + entry["content"]).lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results]
