import os
import shutil
import glob
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root directory (assuming preprocess.py is in the src directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Define relative paths
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EYES_OPEN_DIR = PROCESSED_DIR / "eyes_open"
EYES_CLOSED_DIR = PROCESSED_DIR / "eyes_closed"

# Create directories
def create_directories():
    """Create necessary directory structure"""
    for dir_path in [PROCESSED_DIR, EYES_OPEN_DIR, EYES_CLOSED_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    logger.info("Directories created successfully")


# Process files for a single subject
def process_subject_files(subject_dir):
    """
    Process files for a single subject, organizing eyes-open and eyes-closed files
    
    Args:
        subject_dir: Path to the subject directory
    
    Returns:
        success_count: Number of files successfully processed
    """
    success_count = 0
    subject_id = subject_dir.name  # e.g., "S001"
    numeric_id = subject_id.replace('S', '').lstrip('0')  # S001 -> 1
    
    # Find eyes-open files (R01.edf)
    eyes_open_files = list(subject_dir.glob("*R01.edf"))
    if eyes_open_files:
        edf_path = eyes_open_files[0]
        annotation_path = str(edf_path) + ".event"
        
        # Create target directory
        target_dir = EYES_OPEN_DIR / f"sub-{numeric_id}"
        target_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy EDF file
        target_edf = target_dir / f"sub-{numeric_id}_eeg.edf"
        shutil.copy2(edf_path, target_edf)

         
        # Copy annotation file if it exists
        if os.path.exists(annotation_path):
            target_annot = target_dir / f"sub-{numeric_id}_eeg.edf.event"
            shutil.copy2(annotation_path, target_annot)
        
        logger.info(f"Copied {edf_path.name} to {target_edf}")
        success_count += 1
    
    # Find eyes-closed files (R02.edf)
    eyes_closed_files = list(subject_dir.glob("*R02.edf"))
    if eyes_closed_files:
        edf_path = eyes_closed_files[0]
        annotation_path = str(edf_path) + ".event"
        
        # Create target directory
        target_dir = EYES_CLOSED_DIR / f"sub-{numeric_id}"
        target_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy EDF file
        target_edf = target_dir / f"sub-{numeric_id}_eeg.edf"
        shutil.copy2(edf_path, target_edf)
        
        # Copy annotation file if it exists
        if os.path.exists(annotation_path):
            target_annot = target_dir / f"sub-{numeric_id}_eeg.edf.event"
            shutil.copy2(annotation_path, target_annot)
        
        logger.info(f"Copied {edf_path.name} to {target_edf}")
        success_count += 1
    
    return success_count

# Main processing function
def process_all_files():
    """Process files for all subjects"""
    create_directories()
    
    total_success = 0
    total_subjects = 0
    
    # Check if raw data directory exists
    if not RAW_DIR.exists():
        logger.error(f"Raw data directory does not exist: {RAW_DIR}")
        logger.error("Please ensure data is placed in the correct location or modify the RAW_DIR path in the script")
        return
    
    # Loop through all subject directories
    subject_dirs = sorted(RAW_DIR.glob("S*"))
    if not subject_dirs:
        logger.warning(f"No subject directories (S*) found in {RAW_DIR}")
        return
        
    for subject_dir in tqdm(subject_dirs):
        total_subjects += 1
        success_count = process_subject_files(subject_dir)
        total_success += success_count
    
    logger.info(f"Preprocessing complete. Processed {total_subjects} subjects, successfully copied {total_success} files.")

# If run as main program
if __name__ == "__main__":
    logger.info("Starting EEG data file organization...")
    logger.info(f"Project root directory: {PROJECT_ROOT}")
    logger.info(f"Raw data directory: {RAW_DIR}")
    logger.info(f"Processed data directory: {PROCESSED_DIR}")
    
    process_all_files()
    logger.info("File organization complete!")
