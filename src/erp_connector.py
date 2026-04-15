"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: erp_connector.py
  Future Improvement #7:
    - Mock ERP system integration (SAP / Oracle simulation)
    - Exposes a clean API interface that would replace the
      CSV-based data loading in production
    - Simulates GET/POST API calls to ERP endpoints
    - In production: swap MockERPClient with real REST client
      using the `requests` library + OAuth2 authentication
=============================================================

PRODUCTION UPGRADE:
  import requests
  class SAPClient:
      BASE_URL = "https://your-sap-host.com/sap/opu/odata"
      def get_stock_levels(self, plant, material):
          return requests.get(
              f"{self.BASE_URL}/MM_INVENTORY_SRV/InventorySet",
              params={"$filter": f"Plant eq '{plant}'"},
              headers={"Authorization": f"Bearer {self.token}"},
          ).json()
"""

import os
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(REPORT_DIR, exist_ok=True)


# ── Mock ERP Response Schemas ─────────────────────────────
class ERPResponse:
    """Simulates an ERP API JSON response envelope."""
    def __init__(self, status: int, data, message: str = "OK"):
        self.status    = status
        self.data      = data
        self.message   = message
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message,
                "timestamp": self.timestamp, "data": self.data}

    def is_success(self) -> bool:
        return 200 <= self.status < 300


# ── Mock ERP Client ────────────────────────────────────────
class MockERPClient:
    """
    Simulates SAP S/4HANA or Oracle SCM REST API.
    All methods add a simulated network latency of 50–200ms.

    In production replace this class body with real HTTP calls:
      self.session = requests.Session()
      self.session.auth = OAuth2BearerToken(token)
      response = self.session.get(endpoint, params=params)
    """

    ERP_NAME = "MockSAP-S4HANA v2023"
    PLANTS   = {
        "S001": "PLANT_MUM",
        "S002": "PLANT_DEL",
        "S003": "PLANT_BLR",
        "S004": "PLANT_PUN",
        "S005": "PLANT_CHN",
    }

    def __init__(self, base_url: str = "https://mock-erp.retail-system.demo",
                 api_key: str = "DEMO-KEY-12345"):
        self.base_url = base_url
        self.api_key  = api_key
        self._connected = False
        print(f"[ERP] MockERPClient initialized: {self.ERP_NAME}")
        print(f"[ERP] Endpoint : {base_url}")

    def connect(self) -> bool:
        """Simulate authentication handshake."""
        print(f"[ERP] Authenticating ...")
        time.sleep(random.uniform(0.05, 0.15))   # simulate latency
        self._connected = True
        print(f"[ERP] Connected  : {self.ERP_NAME}")
        return True

    def _check_connection(self):
        if not self._connected:
            raise ConnectionError("ERP not connected. Call connect() first.")

    def _simulate_latency(self):
        time.sleep(random.uniform(0.03, 0.10))

    # ── GET: Current Stock Levels ─────────────────────────
    def get_stock_levels(self, store_id: str = None) -> ERPResponse:
        """
        Simulate GET /api/v1/inventory/stock-levels
        Returns current on-hand stock per product per plant.
        """
        self._check_connection()
        self._simulate_latency()

        stores = [store_id] if store_id else list(self.PLANTS.keys())
        products = [f"P{str(i).zfill(3)}" for i in range(1, 16)]
        np.random.seed(int(datetime.now().timestamp()) % 1000)

        records = []
        for sid in stores:
            plant = self.PLANTS.get(sid, "UNKNOWN")
            for pid in products:
                records.append({
                    "plant":             plant,
                    "store_id":          sid,
                    "material_id":       pid,
                    "unrestricted_stock":int(np.random.randint(0, 500)),
                    "in_transit_qty":    int(np.random.randint(0, 100)),
                    "reserved_qty":      int(np.random.randint(0, 50)),
                    "last_updated":      datetime.now().isoformat(),
                    "unit":              "EA",
                })

        return ERPResponse(200, records, "Stock levels retrieved successfully")

    # ── GET: Open Purchase Orders ─────────────────────────
    def get_open_purchase_orders(self) -> ERPResponse:
        """Simulate GET /api/v1/procurement/purchase-orders?status=OPEN"""
        self._check_connection()
        self._simulate_latency()

        orders = []
        for i in range(1, 16):
            expected = datetime.now() + timedelta(days=random.randint(1, 14))
            orders.append({
                "po_number":      f"PO-{random.randint(100000,999999)}",
                "material_id":    f"P{str(random.randint(1,15)).zfill(3)}",
                "store_id":       random.choice(list(self.PLANTS.keys())),
                "quantity":       random.randint(50, 500),
                "unit_price":     round(random.uniform(10, 350), 2),
                "status":         random.choice(["AWAITING_APPROVAL",
                                                  "APPROVED", "IN_TRANSIT"]),
                "expected_delivery": expected.date().isoformat(),
                "supplier":       random.choice(["HUL", "ITC", "Nestle",
                                                  "Dabur", "P&G India"]),
            })
        return ERPResponse(200, orders, "Purchase orders retrieved")

    # ── POST: Create Purchase Order ────────────────────────
    def create_purchase_order(self, store_id: str, product_id: str,
                               quantity: int, unit_price: float,
                               supplier: str = "AUTO-SELECT") -> ERPResponse:
        """
        Simulate POST /api/v1/procurement/purchase-orders
        Creates a new purchase order in the ERP.
        """
        self._check_connection()
        self._simulate_latency()

        po_number = f"PO-{random.randint(100000, 999999)}"
        delivery  = datetime.now() + timedelta(days=random.randint(3, 10))
        total     = round(quantity * unit_price, 2)

        po = {
            "po_number":       po_number,
            "store_id":        store_id,
            "plant":           self.PLANTS.get(store_id, "UNKNOWN"),
            "material_id":     product_id,
            "quantity":        quantity,
            "unit_price":      unit_price,
            "total_value":     total,
            "supplier":        supplier,
            "status":          "AWAITING_APPROVAL",
            "created_at":      datetime.now().isoformat(),
            "expected_delivery": delivery.date().isoformat(),
        }
        print(f"[ERP] PO Created: {po_number} | {product_id} | Qty={quantity}"
              f" | Rs {total:,.0f}")
        return ERPResponse(201, po, f"Purchase order {po_number} created")

    # ── POST: Batch auto-create POs from reorder alerts ───
    def auto_create_pos_from_alerts(self, alerts_df: pd.DataFrame) -> list:
        """
        Loop through reorder alerts and create a PO for each.
        Returns list of ERPResponse objects.
        """
        self._check_connection()
        responses = []
        for _, row in alerts_df.iterrows():
            resp = self.create_purchase_order(
                store_id   = row.get("store_id",    "S001"),
                product_id = row.get("product_id",  "P001"),
                quantity   = int(row.get("recommended_order_qty", 100)),
                unit_price = float(row.get("base_price", 50)),
                supplier   = "AUTO-SELECT",
            )
            responses.append(resp)
        return responses

    # ── GET: Sales transactions (replaces CSV) ─────────────
    def get_sales_data(self, from_date: str, to_date: str,
                        store_id: str = None) -> ERPResponse:
        """
        Simulate GET /api/v1/sales/transactions
        In production this replaces reading from CSV directly.
        """
        self._check_connection()
        self._simulate_latency()

        data = {
            "from_date":      from_date,
            "to_date":        to_date,
            "store_filter":   store_id or "ALL",
            "record_count":   random.randint(8000, 15000),
            "data_format":    "JSON",
            "download_url":   f"{self.base_url}/exports/sales_extract.csv",
            "note": ("In production this returns the actual transaction "
                     "records. In simulation mode, use the CSV files generated "
                     "by data_generator.py as a drop-in replacement."),
        }
        return ERPResponse(200, data, "Sales data endpoint reached")


# ── ERP Data Sync simulation ──────────────────────────────
def simulate_erp_sync(alerts_df: pd.DataFrame = None) -> dict:
    """
    Simulate a full ERP data sync cycle:
      1. Connect
      2. Pull stock levels
      3. Pull open POs
      4. Auto-create POs for reorder alerts
      5. Save sync report
    """
    print("\n" + "=" * 60)
    print("  Running ERP Integration Simulation ...")
    print("=" * 60)

    client = MockERPClient()
    client.connect()

    # Pull stock
    stock_resp = client.get_stock_levels()
    stock_df   = pd.DataFrame(stock_resp.data)
    print(f"[ERP] Stock records received: {len(stock_df)}")

    # Pull POs
    po_resp   = client.get_open_purchase_orders()
    po_df     = pd.DataFrame(po_resp.data)
    print(f"[ERP] Open POs received: {len(po_df)}")

    # Auto-create POs from alerts
    if alerts_df is not None and not alerts_df.empty:
        po_responses = client.auto_create_pos_from_alerts(alerts_df)
        new_pos      = [r.data for r in po_responses if r.is_success()]
        new_pos_df   = pd.DataFrame(new_pos)
        print(f"[ERP] New POs created  : {len(new_pos_df)}")
    else:
        new_pos_df = pd.DataFrame()

    # Sales sync info
    sales_resp = client.get_sales_data("2023-01-01", "2023-12-31")
    print(f"[ERP] Sales data endpoint: {sales_resp.message}")

    # Save sync report
    stock_df.to_csv(os.path.join(REPORT_DIR, "erp_stock_snapshot.csv"), index=False)
    po_df.to_csv(   os.path.join(REPORT_DIR, "erp_open_pos.csv"),        index=False)
    if not new_pos_df.empty:
        new_pos_df.to_csv(os.path.join(REPORT_DIR, "erp_new_pos.csv"),   index=False)

    print(f"[✓] ERP sync complete. Reports saved to reports/")

    return {
        "stock":     stock_df,
        "open_pos":  po_df,
        "new_pos":   new_pos_df,
    }


if __name__ == "__main__":
    alerts_path = os.path.join(os.path.dirname(__file__), "..",
                               "outputs", "reorder_alerts.csv")
    alerts = pd.read_csv(alerts_path) if os.path.exists(alerts_path) else None
    simulate_erp_sync(alerts)
