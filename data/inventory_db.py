import sqlite3
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

ITEM_ALIASES = {
    "oxygen": "o2 canister",
    "oxygen can": "o2 canister",
    "oxygen cans": "o2 canister",
    "oxygen canister": "o2 canister",
    "oxygen canisters": "o2 canister",
    "o2": "o2 canister",
    "o2 cans": "o2 canister",
    "o2 tank": "o2 canister",
    "nitrogen": "n2 tank",
    "nitrogen tank": "n2 tank",
    "n2": "n2 tank",
    "co2": "co2 scrubber",
    "co2 scrubber": "co2 scrubber",
    "scrubber": "co2 scrubber",
    "scrubbers": "co2 scrubber",
    "air filter": "air filter",
    "filter": "water filter",
    "filters": "water filter",
    "water": "water filter",
    "water filter": "water filter",
    "food": "food ration",
    "rations": "food ration",
    "ration": "food ration",
    "med kit": "medical kit",
    "medkit": "medical kit",
    "medical kit": "medical kit",
    "first aid": "medical kit",
    "antibiotic": "antibiotic pack",
    "antibiotics": "antibiotic pack",
    "wrench": "wrench set",
    "wrenches": "wrench set",
    "tools": "wrench set",
    "patch kit": "eva suit patch kit",
    "eva kit": "eva suit patch kit",
}


@dataclass
class ItemRecord:
    id: int
    name: str
    category: str
    unit: str
    quantity: int
    max_quantity: int
    location: str
    criticality: str
    consumption_rate: float
    days_remaining: float
    status: str


class HabitatDatabase:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS inventory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    quantity INTEGER NOT NULL DEFAULT 0,
                    max_quantity INTEGER NOT NULL DEFAULT 100,
                    location TEXT NOT NULL,
                    criticality TEXT CHECK(criticality IN ('critical', 'high', 'medium', 'low')) DEFAULT 'medium',
                    consumption_rate REAL DEFAULT 0.5,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity_change INTEGER DEFAULT 0,
                    performed_by TEXT DEFAULT 'system',
                    location TEXT NOT NULL,
                    mission_day INTEGER DEFAULT 1,
                    parent_hash TEXT DEFAULT 'GENESIS',
                    merkle_hash TEXT DEFAULT 'GENESIS',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_inventory_name ON inventory(name);
                CREATE INDEX IF NOT EXISTS idx_inventory_category ON inventory(category);
                CREATE INDEX IF NOT EXISTS idx_inventory_location ON inventory(location);
            """)

    def seed_initial_data(self, json_path: Optional[str] = None):
        if json_path and Path(json_path).exists():
            with open(json_path) as f:
                data = json.load(f)
                items = data.get("items", [])
        else:
            items = [
                {"name": "o2 canister", "category": "life_support", "unit": "units", "quantity": 48, "max_quantity": 50, "location": "storage bay a1", "criticality": "critical", "consumption_rate": 2.0},
                {"name": "n2 tank", "category": "life_support", "unit": "units", "quantity": 12, "max_quantity": 15, "location": "storage bay a2", "criticality": "critical", "consumption_rate": 0.5},
                {"name": "co2 scrubber", "category": "life_support", "unit": "units", "quantity": 24, "max_quantity": 25, "location": "module alpha locker 1", "criticality": "critical", "consumption_rate": 1.0},
                {"name": "air filter", "category": "life_support", "unit": "units", "quantity": 36, "max_quantity": 40, "location": "module beta locker 1", "criticality": "high", "consumption_rate": 0.8},
                {"name": "water filter", "category": "life_support", "unit": "units", "quantity": 20, "max_quantity": 25, "location": "storage bay b1", "criticality": "critical", "consumption_rate": 0.4},
                {"name": "food ration", "category": "nutrition", "unit": "packets", "quantity": 720, "max_quantity": 1000, "location": "galley storage 1", "criticality": "critical", "consumption_rate": 12.0},
                {"name": "medical kit", "category": "medical", "unit": "kits", "quantity": 6, "max_quantity": 10, "location": "medical bay cabinet 1", "criticality": "critical", "consumption_rate": 0.1},
                {"name": "antibiotic pack", "category": "medical", "unit": "packs", "quantity": 24, "max_quantity": 30, "location": "medical bay cabinet 2", "criticality": "critical", "consumption_rate": 0.2},
                {"name": "wrench set", "category": "tools", "unit": "sets", "quantity": 4, "max_quantity": 5, "location": "module gamma shelf 1", "criticality": "medium", "consumption_rate": 0.01},
                {"name": "eva suit patch kit", "category": "eva", "unit": "kits", "quantity": 6, "max_quantity": 8, "location": "airlock compartment 1", "criticality": "critical", "consumption_rate": 0.05},
            ]

        with self.conn:
            for item in items:
                self.conn.execute("""
                    INSERT OR REPLACE INTO inventory
                    (name, category, unit, quantity, max_quantity, location, criticality, consumption_rate)
                    VALUES (:name, :category, :unit, :quantity, :max_quantity, :location, :criticality, :consumption_rate)
                """, item)

    def _get_latest_merkle_hash(self) -> str:
        row = self.conn.execute("SELECT merkle_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        return row["merkle_hash"] if row else "GENESIS"

    def query_item(self, item_name: str) -> Optional[ItemRecord]:
        item_name_clean = item_name.strip().lower()

        # Check alias dictionary first
        for alias, target in ITEM_ALIASES.items():
            if alias in item_name_clean:
                item_name_clean = target
                break

        row = self.conn.execute(
            "SELECT * FROM inventory WHERE LOWER(name) = ?", (item_name_clean,)
        ).fetchone()

        if not row:
            row = self.conn.execute(
                "SELECT * FROM inventory WHERE LOWER(name) LIKE ?", (f"%{item_name_clean}%",)
            ).fetchone()

        if not row:
            # Token search fallback
            words = [w for w in item_name_clean.split() if len(w) > 2]
            for w in words:
                row = self.conn.execute(
                    "SELECT * FROM inventory WHERE LOWER(name) LIKE ?", (f"%{w}%",)
                ).fetchone()
                if row:
                    break

        if row:
            return self._row_to_record(row)
        return None

    def update_quantity(
        self, item_name: str, delta: int, performed_by: str = "system", mission_day: int = 1
    ) -> Optional[ItemRecord]:
        record = self.query_item(item_name)
        if not record:
            return None

        new_qty = max(0, record.quantity + delta)
        parent_hash = self._get_latest_merkle_hash()
        action = "restock" if delta > 0 else "consume"

        payload = f"{parent_hash}:{record.name}:{action}:{delta}:{performed_by}:{mission_day}"
        merkle_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self.conn:
            self.conn.execute(
                "UPDATE inventory SET quantity = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_qty, record.id),
            )
            self.conn.execute("""
                INSERT INTO audit_log (item_name, action, quantity_change, performed_by, location, mission_day, parent_hash, merkle_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record.name, action, delta, performed_by, record.location, mission_day, parent_hash, merkle_hash))

        return self.query_item(record.name)

    def verify_merkle_chain(self) -> bool:
        rows = self.conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
        expected_parent = "GENESIS"

        for row in rows:
            if row["parent_hash"] != expected_parent:
                return False
            payload = f"{row['parent_hash']}:{row['item_name']}:{row['action']}:{row['quantity_change']}:{row['performed_by']}:{row['mission_day']}"
            computed_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if computed_hash != row["merkle_hash"]:
                return False
            expected_parent = row["merkle_hash"]

        return True

    def locate_item(self, item_name: str) -> Optional[Dict[str, Any]]:
        record = self.query_item(item_name)
        if record:
            return {
                "item": record.name,
                "location": record.location,
                "quantity": record.quantity,
                "unit": record.unit,
                "status": record.status,
            }
        return None

    def get_alerts(self, threshold_ratio: float = 0.25) -> List[ItemRecord]:
        rows = self.conn.execute("SELECT * FROM inventory").fetchall()
        alerts = []
        for row in rows:
            rec = self._row_to_record(row)
            ratio = row["quantity"] / max(row["max_quantity"], 1)
            if ratio <= threshold_ratio:
                alerts.append(rec)
        return alerts

    def log_maintenance(
        self, action: str, item_name: str, location: str, performed_by: str = "engineer", mission_day: int = 1
    ) -> Dict[str, Any]:
        parent_hash = self._get_latest_merkle_hash()
        payload = f"{parent_hash}:{item_name}:{action}:0:{performed_by}:{mission_day}"
        merkle_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self.conn:
            self.conn.execute("""
                INSERT INTO audit_log (item_name, action, quantity_change, performed_by, location, mission_day, parent_hash, merkle_hash)
                VALUES (?, ?, 0, ?, ?, ?, ?, ?)
            """, (item_name, action, performed_by, location, mission_day, parent_hash, merkle_hash))

        return {
            "status": "success",
            "item": item_name,
            "action": action,
            "location": location,
            "performed_by": performed_by,
            "mission_day": mission_day,
            "merkle_hash": merkle_hash,
        }

    def _row_to_record(self, row: sqlite3.Row) -> ItemRecord:
        qty = row["quantity"]
        max_qty = row["max_quantity"]
        rate = row["consumption_rate"]
        ratio = qty / max(max_qty, 1)

        if ratio <= 0:
            status = "empty"
        elif ratio <= 0.1:
            status = "critical"
        elif ratio <= 0.25:
            status = "low"
        elif ratio <= 0.75:
            status = "nominal"
        else:
            status = "full"

        days_remaining = round(qty / rate, 1) if rate > 0 else 999.0

        return ItemRecord(
            id=row["id"],
            name=row["name"],
            category=row["category"],
            unit=row["unit"],
            quantity=qty,
            max_quantity=max_qty,
            location=row["location"],
            criticality=row["criticality"],
            consumption_rate=rate,
            days_remaining=days_remaining,
            status=status,
        )

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
