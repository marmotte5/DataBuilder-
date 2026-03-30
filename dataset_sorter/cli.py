"""
Module: cli.py
==============
Command-line interface for DataBuilder.

Subcommands:
    serve-mcp   Start the stdio MCP server for AI agent integration.

Usage:
    dataset-sorter-mcp          # via pyproject.toml entry point
    python -m dataset_sorter serve-mcp
"""

import argparse
import logging
import sys


def _cmd_serve_mcp(_args: argparse.Namespace) -> None:
    """Start the stdio MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Keep stderr for logs so stdout stays clean JSON-RPC
    )
    from dataset_sorter.mcp_server import run_server
    run_server()


def main(argv: list[str] | None = None) -> None:
    """Entry point for the DataBuilder CLI."""
    parser = argparse.ArgumentParser(
        prog="dataset-sorter",
        description="DataBuilder — high-performance text-to-image training and generation.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # serve-mcp subcommand
    sub.add_parser(
        "serve-mcp",
        help="Start the stdio MCP server for AI agent integration (Claude Desktop, etc.).",
    )

    args = parser.parse_args(argv)

    if args.command == "serve-mcp":
        _cmd_serve_mcp(args)
    else:
        # No subcommand: launch the GUI (existing behaviour)
        from dataset_sorter.startup_log import print_startup_log
        from dataset_sorter.ui.main_window import run
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        print_startup_log()
        run()


if __name__ == "__main__":
    main()
