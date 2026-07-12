"""
Sprint 6C.7 — Interactive CLI Demo Menu (simplified).

Run with:
    python cli.py

This is an optional demo entry point for interview and local exploration.
Running python main.py is unchanged and unaffected by this file.

The menu loads config.yaml on startup.  The embedding model is loaded lazily
on first use of an option that requires it (options 1, 2, 3).  All other
options (DB queries, config display, leakage audit) work without it.

Six top-level options:
    0. Exit
    1. Triage a ticket
    2. Run batch evaluation
    3. Compare reviewer OFF vs ON
    4. Inspect evaluation results
    5. Run leakage audit
    6. Show runtime configuration
"""

import os
import sys

import yaml

from src.application.cli_menu import (
    DB_PATH,
    apply_reviewer_override,
    load_embedding_model,
    lookup_prediction,
    lookup_ticket,
    print_effective_run_config,
    run_batch_evaluation,
    run_leakage_audit,
    run_random_eval_ticket,
    run_reviewer_ab_comparison,
    run_specific_ticket,
    show_config,
    show_confusion_matrix,
    show_curated_leaderboard,
    show_kpi_leaderboard,
    show_run_details,
)

CONFIG_PATH = "config.yaml"

MENU = """\

╔══════════════════════════════════════════════════════════╗
║         HDI Ticket Triage Agent — Demo Menu              ║
╠══════════════════════════════════════════════════════════╣
║  0.  Exit                                                ║
║  1.  Triage a ticket                                     ║
║  2.  Run batch evaluation                                ║
║  3.  Compare reviewer OFF vs ON                          ║
║  4.  Inspect evaluation results                          ║
║  5.  Run leakage audit                                   ║
║  6.  Show runtime configuration                          ║
╚══════════════════════════════════════════════════════════╝"""

_TRIAGE_MENU = """\

  ╔══════════════════════════════════════════════════════╗
  ║                  Triage a Ticket                     ║
  ╠══════════════════════════════════════════════════════╣
  ║  0.  Back                                            ║
  ║  1.  Random eval ticket                              ║
  ║  2.  Specific ticket_id                              ║
  ╚══════════════════════════════════════════════════════╝"""

_EVAL_MENU = """\

  ╔══════════════════════════════════════════════════════╗
  ║            Inspect Evaluation Results                ║
  ╠══════════════════════════════════════════════════════╣
  ║  0.  Back                                            ║
  ║  1.  Curated evaluation                              ║
  ║  2.  Full experiment history                         ║
  ║  3.  Run details                                     ║
  ║  4.  Confusion matrix                                ║
  ║  5.  Ticket prediction details                       ║
  ╚══════════════════════════════════════════════════════╝"""


# ── Input helpers ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prompt(label: str) -> str:
    """Read a line from stdin with a label. Strips whitespace."""
    try:
        return input(f"  {label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _prompt_run_id() -> str:
    return _prompt("Enter run_id (e.g. run_20260710_143022)")


def _prompt_ticket_id() -> str:
    return _prompt("Enter ticket_id (16-char hex)")


# ── Reviewer mode selection ───────────────────────────────────────────────────

def _select_reviewer_mode(cfg: dict) -> dict | None:
    """
    Ask user to select the reviewer mode for a single operation.

    Options:
      1. Analyzer only      — reviewer.enabled = False in the returned copy
      2. Conditional reviewer — reviewer.enabled = True in the returned copy
      0. Cancel / back      — returns None

    Requirements:
      - Never writes config.yaml.
      - Returns a deep copy via apply_reviewer_override; original cfg is untouched.
      - If "Conditional reviewer" is selected but reviewer.model_name is not
        configured, prints an explanation and returns None.
    """
    print("\n  ── Reviewer Mode for This Operation ──────────")
    print("  1.  Analyzer only")
    print("  2.  Conditional reviewer")
    print("  0.  Cancel")
    choice = _prompt("Select reviewer mode")

    if choice == "0" or not choice:
        return None

    if choice == "1":
        return apply_reviewer_override(cfg, False)

    if choice == "2":
        reviewer_model = cfg.get("reviewer", {}).get("model_name", "")
        if not reviewer_model:
            print("  No reviewer model is configured in config.yaml.")
            print("  Set reviewer.model_name under the 'reviewer' section to use it.")
            return None
        return apply_reviewer_override(cfg, True)

    print(f"  Invalid choice '{choice}'. Returning to previous menu.")
    return None


# ── Submenu: Triage a ticket ──────────────────────────────────────────────────

def _triage_ticket_menu(
    cfg: dict,
    db_path: str,
    embedding_model,
):
    """
    Submenu for option 1: Triage a ticket.

    Offers random eval ticket or specific ticket_id.
    Asks for reviewer mode before each triage operation.

    Returns the embedding model (possibly loaded for the first time).
    """
    while True:
        print(_TRIAGE_MENU)
        choice = _prompt("Select an option")

        if choice == "0":
            break

        elif choice == "1":
            effective_cfg = _select_reviewer_mode(cfg)
            if effective_cfg is None:
                continue
            if embedding_model is None:
                embedding_model = load_embedding_model(effective_cfg)
            run_random_eval_ticket(effective_cfg, db_path, embedding_model)

        elif choice == "2":
            ticket_id = _prompt_ticket_id()
            if not ticket_id:
                print("  No ticket_id provided.")
                continue
            effective_cfg = _select_reviewer_mode(cfg)
            if effective_cfg is None:
                continue
            if embedding_model is None:
                embedding_model = load_embedding_model(effective_cfg)
            run_specific_ticket(effective_cfg, db_path, ticket_id, embedding_model)

        else:
            print(f"  Unknown option '{choice}'. Enter 0, 1, or 2.")

    return embedding_model


# ── Submenu: Inspect evaluation results ──────────────────────────────────────

def _ticket_prediction_details(db_path: str) -> None:
    """
    Show original ticket info then the stored prediction for a ticket_id + run_id.

    Calls lookup_ticket and lookup_prediction sequentially without duplicating SQL.
    Handles a missing ticket or run gracefully (lookup functions print their own
    error messages).
    """
    ticket_id = _prompt_ticket_id()
    if not ticket_id:
        print("  No ticket_id provided.")
        return
    lookup_ticket(db_path, ticket_id)

    run_id = _prompt_run_id()
    if not run_id:
        print("  No run_id provided. Ticket info shown above.")
        return
    lookup_prediction(db_path, ticket_id, run_id)


def _evaluation_results_menu(cfg: dict, db_path: str) -> None:
    """
    Submenu for option 4: Inspect evaluation results.

    Options:
      1. Curated evaluation    — grouped curated view from config.yaml
      2. Full experiment history — existing unfiltered leaderboard
      3. Run details
      4. Confusion matrix
      5. Ticket prediction details (original ticket + stored prediction)
    """
    while True:
        print(_EVAL_MENU)
        choice = _prompt("Select an option")

        if choice == "0":
            break

        elif choice == "1":
            show_curated_leaderboard(db_path, cfg)

        elif choice == "2":
            show_kpi_leaderboard(db_path)

        elif choice == "3":
            run_id = _prompt_run_id()
            if not run_id:
                print("  No run_id provided.")
                continue
            show_run_details(db_path, run_id)

        elif choice == "4":
            run_id = _prompt_run_id()
            if not run_id:
                print("  No run_id provided.")
                continue
            show_confusion_matrix(db_path, run_id)

        elif choice == "5":
            _ticket_prediction_details(db_path)

        else:
            print(f"  Unknown option '{choice}'. Enter 0–5.")


# ── Main demo loop ────────────────────────────────────────────────────────────

def _demo_loop(cfg: dict, db_path: str, lance_path: str) -> None:
    """
    Core interactive loop.

    Separated from main() so tests can call it directly with a pre-built config
    without requiring config.yaml to exist on disk.

    The embedding model is loaded lazily — only when a triage or A/B option is
    selected for the first time.
    """
    embedding_model = None

    while True:
        print(MENU)
        choice = _prompt("Select an option")

        if choice == "0":
            print("  Exiting.")
            break

        elif choice == "1":
            embedding_model = _triage_ticket_menu(cfg, db_path, embedding_model)

        elif choice == "2":
            raw = _prompt("Enter limit (number of tickets, or press Enter for config default)")
            batch_limit = cfg.get("batch", {}).get("limit", 20)
            if raw.isdigit():
                batch_limit = int(raw)
            elif raw:
                print(f"  Invalid limit '{raw}'. Using config default: {batch_limit}.")
            print(f"  Using limit: {batch_limit}")
            effective_cfg = _select_reviewer_mode(cfg)
            if effective_cfg is None:
                continue
            print_effective_run_config(effective_cfg, batch_limit)
            if embedding_model is None:
                embedding_model = load_embedding_model(effective_cfg)
            run_batch_evaluation(effective_cfg, db_path, batch_limit, embedding_model)

        elif choice == "3":
            if embedding_model is None:
                embedding_model = load_embedding_model(cfg)
            run_reviewer_ab_comparison(cfg, db_path, embedding_model)

        elif choice == "4":
            _evaluation_results_menu(cfg, db_path)

        elif choice == "5":
            run_leakage_audit(db_path, lance_path)

        elif choice == "6":
            show_config(cfg)

        else:
            print(f"  Unknown option '{choice}'. Enter a number 0–6.")


def main() -> None:
    """
    Entry point for the interactive CLI demo.

    Loads config.yaml, prints startup info, then runs the interactive loop.
    Running python main.py is unchanged and unaffected by this file.
    """
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} not found. Run from the project root.")
        sys.exit(1)

    cfg = _load_config()
    lance_path = cfg.get("vector_store", {}).get("path", "data/lancedb")

    print(f"\nTicket Triage Agent CLI — config loaded from {CONFIG_PATH}")
    print(f"DuckDB: {DB_PATH}")
    print(f"LanceDB: {lance_path}")

    _demo_loop(cfg, DB_PATH, lance_path)


if __name__ == "__main__":
    main()
