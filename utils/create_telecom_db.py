import sqlite3
from pathlib import Path

DB_PATH = Path("telecom_support.db")


def create_database(db_path: Path) -> None:
    """Create the telecom_support.db SQLite database with one table and sample data."""
    # Remove existing file if re-running so the script stays idempotent
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 1. Create the support_tickets table (only essential columns)
        cursor.execute(
            """
            CREATE TABLE support_tickets (
                ticket_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                city TEXT NOT NULL,
                issue_category TEXT NOT NULL,
                issue_description TEXT NOT NULL,
                priority TEXT NOT NULL,
                ticket_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                assigned_team TEXT NOT NULL
            );
            """
        )

        # 2. Add basic indexes for common filter and grouping columns
        cursor.execute("CREATE INDEX idx_tickets_status ON support_tickets(ticket_status);")
        cursor.execute("CREATE INDEX idx_tickets_priority ON support_tickets(priority);")
        cursor.execute("CREATE INDEX idx_tickets_city ON support_tickets(city);")
        cursor.execute("CREATE INDEX idx_tickets_category ON support_tickets(issue_category);")
        cursor.execute("CREATE INDEX idx_tickets_team ON support_tickets(assigned_team);")

        # 3. Insert 12 realistic fictional telecom support tickets across Indian cities
        sample_tickets = [
            (
                "TKT-1001",
                "CUST-201",
                "Aarav Sharma",
                "Bengaluru",
                "Billing",
                "Charged twice for monthly postpaid plan on November invoice.",
                "High",
                "Open",
                "2026-11-18",
                "Billing & Payments",
            ),
            (
                "TKT-1002",
                "CUST-202",
                "Priya Nair",
                "Bengaluru",
                "Broadband",
                "Fiber internet disconnects every 30 minutes in Indiranagar.",
                "High",
                "Open",
                "2026-11-19",
                "Broadband Support",
            ),
            (
                "TKT-1003",
                "CUST-203",
                "Rohan Kulkarni",
                "Mumbai",
                "Broadband",
                "Download speed dropped to 5 Mbps on a 200 Mbps fiber plan.",
                "Medium",
                "In Progress",
                "2026-11-19",
                "Broadband Support",
            ),
            (
                "TKT-1004",
                "CUST-204",
                "Ananya Reddy",
                "Hyderabad",
                "Network",
                "Complete 5G mobile signal outage in Madhapur since morning.",
                "Critical",
                "Open",
                "2026-11-20",
                "Network Operations",
            ),
            (
                "TKT-1005",
                "CUST-205",
                "Vikram Iyer",
                "Chennai",
                "SIM",
                "eSIM QR code activation failing with invalid profile error.",
                "Medium",
                "In Progress",
                "2026-11-20",
                "SIM & Activation",
            ),
            (
                "TKT-1006",
                "CUST-206",
                "Sneha Deshmukh",
                "Pune",
                "Recharge",
                "Prepaid Rs. 599 pack debited via UPI but validity not updated.",
                "High",
                "Open",
                "2026-11-21",
                "Recharge Support",
            ),
            (
                "TKT-1007",
                "CUST-207",
                "Karthik Rao",
                "Bengaluru",
                "Network",
                "Frequent call drops and zero LTE data indoors in Whitefield.",
                "Critical",
                "In Progress",
                "2026-11-21",
                "Network Operations",
            ),
            (
                "TKT-1008",
                "CUST-208",
                "Meera Joshi",
                "Mumbai",
                "SIM",
                "Requested physical SIM replacement after phone theft; awaiting activation.",
                "High",
                "Open",
                "2026-11-22",
                "SIM & Activation",
            ),
            (
                "TKT-1009",
                "CUST-209",
                "Siddharth Verma",
                "Hyderabad",
                "Billing",
                "International roaming add-on billed despite cancellation before travel.",
                "Medium",
                "Resolved",
                "2026-11-15",
                "Billing & Payments",
            ),
            (
                "TKT-1010",
                "CUST-210",
                "Divya Menon",
                "Chennai",
                "Broadband",
                "Router LOS red light blinking after nearby street cabling work.",
                "Medium",
                "Resolved",
                "2026-11-16",
                "Broadband Support",
            ),
            (
                "TKT-1011",
                "CUST-211",
                "Aditya Patel",
                "Mumbai",
                "Recharge",
                "Annual prepaid data voucher missing 2 GB daily quota benefit.",
                "Low",
                "Closed",
                "2026-11-12",
                "Recharge Support",
            ),
            (
                "TKT-1012",
                "CUST-212",
                "Neha Kapoor",
                "Bengaluru",
                "Billing",
                "GST number missing on corporate postpaid bill PDF.",
                "Low",
                "Closed",
                "2026-11-14",
                "Billing & Payments",
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO support_tickets (
                ticket_id, customer_id, customer_name, city,
                issue_category, issue_description, priority,
                ticket_status, created_at, assigned_team
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            sample_tickets,
        )
        conn.commit()


def inspect_and_test_database(db_path: Path) -> None:
    """Print table schema, total record count, and run example RAG test queries."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        print("=== Table Schema: support_tickets ===")
        cursor.execute("PRAGMA table_info(support_tickets);")
        for col in cursor.fetchall():
            # col = (cid, name, type, notnull, dflt_value, pk)
            print(f"  {col[1]} ({col[2]}) | NOT NULL={bool(col[3])} | PK={bool(col[5])}")

        cursor.execute("SELECT COUNT(*) FROM support_tickets;")
        total_records = cursor.fetchone()[0]
        print(f"\nTotal records: {total_records}\n")

        queries = [
            (
                "1. How many tickets are currently open?",
                "SELECT COUNT(*) AS open_tickets FROM support_tickets WHERE ticket_status = 'Open';",
            ),
            (
                "2. How many critical-priority tickets exist?",
                "SELECT COUNT(*) AS critical_tickets FROM support_tickets WHERE priority = 'Critical';",
            ),
            (
                "3. Which city has the most support tickets?",
                """
                SELECT city, COUNT(*) AS ticket_count
                FROM support_tickets
                GROUP BY city
                ORDER BY ticket_count DESC
                LIMIT 1;
                """,
            ),
            (
                "4. Show all unresolved broadband complaints.",
                """
                SELECT ticket_id, customer_name, city, ticket_status, issue_description
                FROM support_tickets
                WHERE issue_category = 'Broadband'
                  AND ticket_status NOT IN ('Resolved', 'Closed');
                """,
            ),
            (
                "5. How many tickets are assigned to the Network Operations team?",
                "SELECT COUNT(*) AS netops_tickets FROM support_tickets WHERE assigned_team = 'Network Operations';",
            ),
            (
                "6. Show all billing-related complaints from Bengaluru.",
                """
                SELECT ticket_id, customer_name, priority, ticket_status, issue_description
                FROM support_tickets
                WHERE issue_category = 'Billing'
                  AND city = 'Bengaluru';
                """,
            ),
        ]

        print("=== Example SQL Queries ===")
        for question, sql in queries:
            print(f"\n{question}")
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                print(f"  {row}")


if __name__ == "__main__":
    create_database(DB_PATH)
    inspect_and_test_database(DB_PATH)