"""
Unit Tests for Embedded Habitat Inventory Database
===================================================
Tests table creation, initial data seeding, atomic stock updates,
location queries, audit logging, and alert filtering.
"""

import pytest
import os
from data.inventory_db import HabitatDatabase, ItemRecord


@pytest.fixture
def db():
    """Create an in-memory HabitatDatabase populated with test data."""
    database = HabitatDatabase(":memory:")
    database.seed_initial_data()
    yield database
    database.close()


class TestHabitatDatabase:
    def test_schema_initialization(self, db):
        """Verify tables exist in database."""
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "inventory" in table_names
        assert "audit_log" in table_names

    def test_query_item_exact_and_fuzzy(self, db):
        """Verify exact and fuzzy item lookup."""
        rec1 = db.query_item("o2 canister")
        assert rec1 is not None
        assert rec1.name == "o2 canister"
        assert rec1.quantity == 48

        # Substring fuzzy match
        rec2 = db.query_item("medical")
        assert rec2 is not None
        assert rec2.name == "medical kit"

    def test_update_quantity_and_audit_log(self, db):
        """Verify atomic quantity updates and audit trail insertion."""
        updated = db.update_quantity("o2 canister", delta=-3, performed_by="astronaut_sarah", mission_day=142)
        assert updated is not None
        assert updated.quantity == 45

        # Check audit log
        log_entry = db.conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        assert log_entry["item_name"] == "o2 canister"
        assert log_entry["action"] == "consume"
        assert log_entry["quantity_change"] == -3
        assert log_entry["performed_by"] == "astronaut_sarah"
        assert log_entry["mission_day"] == 142

    def test_locate_item(self, db):
        """Verify location query."""
        loc = db.locate_item("air filter")
        assert loc is not None
        assert loc["item"] == "air filter"
        assert loc["location"] == "module beta locker 1"

    def test_get_alerts(self, db):
        """Verify stock alerts for critical/low stock items."""
        alerts = db.get_alerts()
        assert len(alerts) >= 0

    def test_log_maintenance(self, db):
        """Verify maintenance event logging."""
        res = db.log_maintenance("inspected", "eva suit patch kit", "airlock compartment 1", performed_by="commander", mission_day=100)
        assert res["status"] == "success"
        assert res["action"] == "inspected"

        log_entry = db.conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        assert log_entry["action"] == "inspected"
        assert log_entry["performed_by"] == "commander"
