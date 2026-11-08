"""Entry point: python -m lingestra.app [--source webcam|video:<path>|landmarks:<path>] ..."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="lingestra.app")
    p.add_argument("--source", default="webcam", help="webcam | video:<path> | landmarks:<path>")
    p.add_argument("--model", default=str(ROOT / "models" / "demo_family.keras"))
    p.add_argument("--send-mode", default="clipboard", choices=["clipboard", "none"])
    p.add_argument("--demo", action="store_true",
                   help="replay a landmarks file in a loop (headless screenshots)")
    p.add_argument("--store-dir", default=None, help="SecureStore directory (default: app-data dir)")
    p.add_argument("--passphrase", default=None, help="SecureStore passphrase (optional)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow, Config
    if args.demo and not args.source.startswith(("landmarks:", "video:")):
        files = sorted((ROOT / "data" / "landmarks").glob("*.npy"))
        args.source = f"landmarks:{files[0]}" if files else args.source
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Lingestra")
    win = MainWindow(Config(source=args.source, model=args.model, send_mode=args.send_mode,
                            loop=args.demo, passphrase=args.passphrase, store_dir=args.store_dir))
    win.show()
    win.start_worker()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
