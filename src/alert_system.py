"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: alert_system.py
  Future Improvement #5:
    - Automated email alerts when reorder threshold crossed
    - Runs in DRY-RUN mode by default (prints instead of sending)
    - Toggle SEND_REAL_EMAIL=True and configure SMTP to actually send
=============================================================

CONFIGURATION:
  Set environment variables (never hardcode passwords):
    ALERT_EMAIL_FROM    = "your.sender@gmail.com"
    ALERT_EMAIL_TO      = "procurement@company.com"
    ALERT_EMAIL_PASS    = "app-specific-password"
    ALERT_SMTP_HOST     = "smtp.gmail.com"
    ALERT_SMTP_PORT     = 587

  For Gmail: enable 2FA and create an App Password.
  For Outlook: use smtp.office365.com, port 587.
"""

import os
import smtplib
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email               import encoders
from datetime            import datetime
import pandas as pd

warnings.filterwarnings("ignore")

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
ALERT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "reorder_alerts.csv")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────
SEND_REAL_EMAIL = False    # Change to True to send real emails

EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "alerts@retail-system.demo")
EMAIL_TO   = os.environ.get("ALERT_EMAIL_TO",   "procurement@retail-system.demo")
EMAIL_PASS = os.environ.get("ALERT_EMAIL_PASS",  "")
SMTP_HOST  = os.environ.get("ALERT_SMTP_HOST",  "smtp.gmail.com")
SMTP_PORT  = int(os.environ.get("ALERT_SMTP_PORT", "587"))


# ── HTML email template ───────────────────────────────────
def build_html_email(alerts_df: pd.DataFrame,
                      inventory_df: pd.DataFrame) -> str:
    """Build a rich HTML email body for procurement alerts."""
    n_reorder   = (inventory_df["inventory_status"] == "REORDER").sum() \
                   if "inventory_status" in inventory_df.columns else len(alerts_df)
    n_overstock = (inventory_df["inventory_status"] == "OVERSTOCK").sum() \
                   if "inventory_status" in inventory_df.columns else 0
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build alert rows
    rows = ""
    for _, row in alerts_df.head(15).iterrows():
        urgency_color = "#e94560" if row.get("recommended_order_qty", 0) > 200 else "#ffd166"
        rows += f"""
        <tr>
          <td style="padding:8px;border-bottom:1px solid #333">{row.get("store_name","")}</td>
          <td style="padding:8px;border-bottom:1px solid #333">{row.get("product_name","")}</td>
          <td style="padding:8px;border-bottom:1px solid #333">{row.get("category","")}</td>
          <td style="padding:8px;border-bottom:1px solid #333;color:#e94560">
            {row.get("current_stock", 0):.0f}</td>
          <td style="padding:8px;border-bottom:1px solid #333">
            {row.get("reorder_point", 0):.0f}</td>
          <td style="padding:8px;border-bottom:1px solid #333;
                     color:{urgency_color};font-weight:bold">
            {row.get("recommended_order_qty", 0):.0f}</td>
          <td style="padding:8px;border-bottom:1px solid #333">
            Rs {row.get("estimated_order_cost", 0):,.0f}</td>
        </tr>"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body {{ font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white;
                      border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.15); }}
        .header {{ background: linear-gradient(135deg, #1a1a2e, #e94560);
                   color: white; padding: 24px; border-radius: 8px 8px 0 0; }}
        .header h1 {{ margin: 0; font-size: 22px; }}
        .header p  {{ margin: 6px 0 0; opacity: 0.85; font-size: 13px; }}
        .kpi-bar {{ display: flex; gap: 16px; padding: 20px; background: #fafafa; }}
        .kpi {{ flex: 1; text-align: center; background: white; padding: 14px;
                border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
        .kpi .num {{ font-size: 28px; font-weight: bold; }}
        .red  {{ color: #e94560; }}
        .yel  {{ color: #f59e0b; }}
        .grn  {{ color: #10b981; }}
        table {{ width: 100%; border-collapse: collapse; margin: 0 20px; width: calc(100% - 40px); }}
        th {{ background: #1a1a2e; color: white; padding: 10px 8px; text-align: left; font-size: 12px; }}
        tr:hover td {{ background: #f9fafb; }}
        .footer {{ padding: 16px 20px; font-size: 11px; color: #999; border-top: 1px solid #eee; }}
      </style>
    </head>
    <body>
    <div class="container">
      <div class="header">
        <h1>&#x1F6A8; Retail Inventory Alert — Action Required</h1>
        <p>Generated: {timestamp} &nbsp;|&nbsp;
           System: Retail Sales Forecasting & Inventory Optimization</p>
      </div>

      <div class="kpi-bar">
        <div class="kpi">
          <div class="num red">{n_reorder}</div>
          <div>Products to REORDER</div>
        </div>
        <div class="kpi">
          <div class="num yel">{n_overstock}</div>
          <div>OVERSTOCK Items</div>
        </div>
        <div class="kpi">
          <div class="num grn">Rs {alerts_df.get("estimated_order_cost",
            pd.Series([0])).sum():,.0f}</div>
          <div>Total Order Value</div>
        </div>
      </div>

      <div style="padding: 0 20px 20px">
        <h3 style="color:#1a1a2e">Priority Reorder List (Top 15)</h3>
        <table>
          <thead>
            <tr>
              <th>Store</th><th>Product</th><th>Category</th>
              <th>Current Stock</th><th>Reorder Point</th>
              <th>Order Qty</th><th>Order Value</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="font-size:12px;color:#666;margin-top:12px">
          * Based on EOQ model with 95% service level (Z=1.65) and
            12-week demand forecast.
        </p>
      </div>

      <div class="footer">
        This is an automated alert from the Retail Intelligence System.
        Please do not reply to this email. Contact your system administrator
        to update alert thresholds or subscription settings.
      </div>
    </div>
    </body>
    </html>"""
    return html


def build_plain_text_alert(alerts_df: pd.DataFrame) -> str:
    """Plain-text version of the alert."""
    lines = [
        "=" * 60,
        "  RETAIL INVENTORY REORDER ALERT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"\n  {len(alerts_df)} products require immediate reorder!\n",
        "-" * 60,
    ]
    for _, row in alerts_df.head(10).iterrows():
        lines.append(
            f"  [{row.get('store_name','')}] {row.get('product_name','')} | "
            f"Stock: {row.get('current_stock',0):.0f} | "
            f"ROP: {row.get('reorder_point',0):.0f} | "
            f"Order: {row.get('recommended_order_qty',0):.0f} units | "
            f"Cost: Rs {row.get('estimated_order_cost',0):,.0f}"
        )
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def save_alert_log(alerts_df: pd.DataFrame) -> str:
    """Save alert to a log file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = os.path.join(REPORT_DIR, "alert_logs")
    os.makedirs(log_dir, exist_ok=True)
    path      = os.path.join(log_dir, f"alert_{timestamp}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(build_plain_text_alert(alerts_df))

    return path


def send_email_alert(alerts_df: pd.DataFrame,
                      inventory_df: pd.DataFrame,
                      attach_csv: bool = True) -> bool:
    """
    Send the reorder alert email.
    If SEND_REAL_EMAIL=False, prints to console (dry-run mode).
    Returns True on success / dry-run, False on error.
    """
    html_body  = build_html_email(alerts_df, inventory_df)
    plain_body = build_plain_text_alert(alerts_df)

    if not SEND_REAL_EMAIL:
        print("\n" + "=" * 60)
        print("  [DRY-RUN] EMAIL ALERT SIMULATION")
        print("=" * 60)
        print(f"  FROM    : {EMAIL_FROM}")
        print(f"  TO      : {EMAIL_TO}")
        print(f"  SUBJECT : URGENT: {len(alerts_df)} Products Need Reorder")
        print("\n" + plain_body)
        print("\n  [DRY-RUN] HTML email body generated. "
              "Set SEND_REAL_EMAIL=True to send.")
        print("=" * 60)
        return True

    # ── Real email sending ─────────────────────────────────
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (f"URGENT: {len(alerts_df)} Products Need Reorder "
                          f"— {datetime.now().strftime('%Y-%m-%d')}")
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO

        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body,  "html"))

        # Attach CSV
        if attach_csv and os.path.exists(ALERT_PATH):
            with open(ALERT_PATH, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                                "attachment",
                                filename="reorder_alerts.csv")
                msg.attach(part)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print(f"[✓] Email alert sent to {EMAIL_TO}")
        return True

    except Exception as e:
        print(f"[!] Failed to send email: {e}")
        return False


def run_alert_system(inventory_df: pd.DataFrame = None,
                     alerts_df:    pd.DataFrame = None) -> bool:
    """
    Main alert runner.
    Loads reorder_alerts.csv if not passed directly.
    """
    print("\n" + "=" * 60)
    print("  Running Alert System ...")
    print("=" * 60)

    # Load data
    if alerts_df is None:
        if os.path.exists(ALERT_PATH):
            alerts_df = pd.read_csv(ALERT_PATH)
        else:
            print("[!] reorder_alerts.csv not found. Run inventory_optimizer first.")
            return False

    if inventory_df is None:
        inv_path = os.path.join(os.path.dirname(__file__), "..",
                                "outputs", "inventory_report.csv")
        inventory_df = pd.read_csv(inv_path) if os.path.exists(inv_path) \
                       else pd.DataFrame()

    if alerts_df.empty:
        print("[✓] No reorder alerts — all inventory levels are OK!")
        return True

    # Save log
    log_path = save_alert_log(alerts_df)
    print(f"[✓] Alert log saved -> {log_path}")

    # Send / simulate email
    success = send_email_alert(alerts_df, inventory_df)
    return success


if __name__ == "__main__":
    run_alert_system()
