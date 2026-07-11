import os
import logging
import sys
from pathlib import Path


from app import config
from app import database

def cleanup_images_for_person(person_id, person_name, dry_run=False):
    """
    Finds the best image for a person, sets it as their profile picture,
    and deletes all other raw and cropped images from disk to save storage.
    """
    images = database.get_images_for_person(person_id)
    
    if not images:
        logging.info(f"[{person_name}] No images found in database.")
        return
        
    # Sort images to find the best one
    # Priority: highest blur_score (sharpest)
    # Ensure it's a valid crop (cropped_width > 0)
    valid_images = [img for img in images if img[4] is not None and img[4] > 0]
    
    if not valid_images:
        logging.info(f"[{person_name}] No valid face crops found.")
        return
        
    # Sort by blur score descending (sharpest first)
    valid_images.sort(key=lambda x: x[3] if x[3] is not None else 0.0, reverse=True)
    
    best_image = valid_images[0]
    best_image_id = best_image[0]
    best_raw_path = best_image[1]
    best_cropped_path = best_image[2]
    
    logging.info(f"[{person_name}] Selected Image ID {best_image_id} as profile image (Blur Score: {best_image[3]:.2f})")
    
    if not dry_run:
        database.set_profile_image_path(person_id, best_cropped_path)
    
    deleted_count = 0
    saved_bytes = 0
    
    for img in images:
        img_id = img[0]
        raw_path = img[1]
        crop_path = img[2]
        
        # Skip the best image entirely so we keep both its raw and cropped versions
        # Or maybe we only keep the cropped version?
        # User requirement: "Keep ONLY one representative profile image. Delete all other registration images from disk."
        # Let's keep both raw and cropped for the best image to be safe.
        if img_id == best_image_id:
            continue
            
        # Delete raw image
        if raw_path:
            full_raw_path = config.PROJECT_ROOT / raw_path
            if full_raw_path.exists():
                saved_bytes += full_raw_path.stat().st_size
                if not dry_run:
                    full_raw_path.unlink()
                deleted_count += 1
                
        # Delete cropped image
        if crop_path:
            full_crop_path = config.PROJECT_ROOT / crop_path
            if full_crop_path.exists():
                saved_bytes += full_crop_path.stat().st_size
                if not dry_run:
                    full_crop_path.unlink()
                deleted_count += 1
                
        # Set paths to NULL in DB to reflect deletion while keeping embeddings valid
        if not dry_run:
            database.nullify_image_paths(img_id)
    
    mb_saved = saved_bytes / (1024 * 1024)
    logging.info(f"[{person_name}] Deleted {deleted_count} files, saving {mb_saved:.2f} MB.")

def main():
    database.init_tables()
    logging.info("=" * 60)
    logging.info("      STORAGE OPTIMIZATION: CLEANUP ENROLLMENT IMAGES")
    logging.info("=" * 60)
    
    if config.KEEP_REGISTRATION_IMAGES:
        logging.info("config.KEEP_REGISTRATION_IMAGES is True. Skipping cleanup.")
        return
        
    persons = database.load_persons()
    if not persons:
        logging.info("No persons enrolled in the database.")
        return
        
    logging.info(f"Found {len(persons)} enrolled persons. Starting cleanup...")
    logging.info("-" * 60)
    
    for p in persons:
        cleanup_images_for_person(p["person_id"], p["name"], dry_run=False)
        
    logging.info("-" * 60)
    logging.info("Storage optimization complete!")

if __name__ == "__main__":
    main()
