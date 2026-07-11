import os
import logging
import shutil
import sqlite3
import argparse
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Setup logging
log_file = Path(__file__).resolve().parent / "attendance.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3),
        logging.StreamHandler()
    ]
)


def main():
    parser = argparse.ArgumentParser(description="Reset deployment data for Raspberry Pi 5 Face Recognition System")
    parser.add_argument("--confirm", action="store_true", help="Confirm deletion of all dataset images and database tables")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset"
    db_path = project_root / "database" / "attendance.db"

    if not args.confirm:
        logging.info("=" * 60)
        logging.info("  DRY RUN - No files will be deleted")
        logging.info("  Run with --confirm to actually reset the deployment")
        logging.info("=" * 60)

    try:
        result = subprocess.run(['systemctl', 'is-active', 'attendance.service'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stdout.strip() == 'active':
            logging.info("WARNING: attendance.service is currently running!")
            logging.info("Please stop the service before resetting the deployment.")
            logging.info("Command: sudo systemctl stop attendance.service")
            return
    except Exception:
        pass
    
    # 1. Clear dataset directories
    logging.info(f"\nScanning dataset directory: {dataset_dir}")
    if dataset_dir.exists():
        dirs_to_delete = []
        for child in dataset_dir.iterdir():
            if child.is_dir() and child.name != "cropped_face":
                dirs_to_delete.append(child)
        
        cropped_dir = dataset_dir / "cropped_face"
        if cropped_dir.exists():
            for child in cropped_dir.iterdir():
                if child.is_dir():
                    dirs_to_delete.append(child)

        for d in dirs_to_delete:
            logging.info(f"  [Target] {d}")
            if args.confirm:
                try:
                    shutil.rmtree(d)
                    logging.info(f"    -> Deleted")
                except Exception as e:
                    logging.info(f"    -> Error deleting: {e}")
    else:
        logging.info("  Dataset directory not found.")

    # 2. Reset Database
    logging.info(f"\nScanning database: {db_path}")
    if db_path.exists():
        logging.info(f"  [Target] {db_path}")
        if args.confirm:
            try:
                conn = sqlite3.connect(str(db_path))
                # Checkpoint and truncate WAL before dropping tables
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception as e:
                    logging.warning(f"    -> Warning checkpointing WAL: {e}")
                
                cur = conn.cursor()
                
                # We drop all tables instead of deleting the file, to avoid permissions issues
                tables = ["recognition_events", "face_embeddings", "images", "persons", "attendance_logs"]
                for table in tables:
                    cur.execute(f"DROP TABLE IF EXISTS {table}")
                
                cur.execute("DELETE FROM sqlite_sequence")
                
                conn.commit()
                conn.execute("VACUUM")
                conn.commit()
                
                # Truncate checkpoint again to flush vacuum and clear WAL space
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
                    
                conn.close()
                logging.info("    -> All tables dropped successfully")

                wal_path = db_path.parent / (db_path.name + "-wal")
                shm_path = db_path.parent / (db_path.name + "-shm")
                for sidecar in [wal_path, shm_path]:
                    if sidecar.exists():
                        try:
                            sidecar.unlink()
                            logging.info(f"    -> Deleted sidecar file {sidecar.name}")
                        except Exception as e:
                            logging.warning(f"    -> Warning deleting sidecar file {sidecar.name} (it may be locked): {e}")
            except Exception as e:
                logging.info(f"    -> Error dropping tables: {e}")
    else:
        logging.info("  Database not found.")

    logging.info("\nReset complete!" if args.confirm else "\nDry run complete!")

    # 3. Clean up the application log file if confirmed
    if args.confirm:
        logging.info("Shutting down logging to clean up log file...")
        logging.shutdown()
        if log_file.exists():
            try:
                # Try to unlink log file
                log_file.unlink()
            except Exception:
                # If locked, truncate it instead
                try:
                    with open(log_file, 'w', encoding='utf-8') as f:
                        pass
                except Exception:
                    pass

if __name__ == "__main__":
    main()
