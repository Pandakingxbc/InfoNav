import os
import shutil
from datetime import datetime


def rotate_file_if_needed(file_path, max_size_mb=5, keep_rotations=5):
    """
    Rotate file if it exceeds max size

    Keeps recent data by:
    1. Moving current file to record.txt.1
    2. Shifting existing rotations: .1 -> .2, .2 -> .3, etc.
    3. Removing oldest rotation if exceeds keep_rotations

    Args:
        file_path: Path to the file to check
        max_size_mb: Maximum file size in MB before rotation
        keep_rotations: Number of rotated files to keep
    """
    if not os.path.exists(file_path):
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        # Shift existing rotations
        for i in range(keep_rotations - 1, 0, -1):
            old_rotation = f"{file_path}.{i}"
            new_rotation = f"{file_path}.{i + 1}"
            if os.path.exists(old_rotation):
                if i + 1 <= keep_rotations:
                    shutil.move(old_rotation, new_rotation)
                else:
                    os.remove(old_rotation)  # Remove oldest

        # Move current file to .1
        first_rotation = f"{file_path}.1"
        shutil.move(file_path, first_rotation)

        # Add timestamp marker to rotated file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(first_rotation, 'r+', encoding='utf-8') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"=== File rotated at {timestamp} (exceeded {max_size_mb}MB) ===\n\n" + content)

        print(f"[Record] Rotated {os.path.basename(file_path)} (was {file_size_mb:.2f}MB)")


def write_record(
    scene_id, episode_id, table, result_text, label, num_total, time_spend, file_path,
    enable_rotation=True, max_size_mb=5
):
    """
    Write navigation episode results to record file with optional rotation

    This function formats and saves navigation episode results including
    performance metrics, success status, and timing information to a
    structured record file for later analysis.

    When file size exceeds max_size_mb, it automatically rotates files:
    - record.txt -> record.txt.1
    - record.txt.1 -> record.txt.2
    - etc.

    This prevents the record file from growing indefinitely while preserving
    all historical data in rotated files.

    Args:
        scene_id: Identifier for the navigation scene
        episode_id: Identifier for the specific episode
        table: Formatted table of performance metrics
        result_text: Success/failure result description
        label: Target object label being searched for
        num_total: Total episode number completed
        time_spend: Time spent on this episode (seconds)
        file_path: Path to the record file to write to
        enable_rotation: Enable file rotation when size limit reached
        max_size_mb: Maximum file size in MB before rotation
    """
    # Rotate file if needed
    if enable_rotation:
        rotate_file_if_needed(file_path, max_size_mb=max_size_mb)

    new_info = f"""
    Scene ID: {scene_id}
    Episode ID: {episode_id}
    {table}
    success or not: {result_text}
    target to find is {label}
    No.{num_total} task is finished
    {time_spend:.2f} seconds spend in this task
    """
    new_info = remove_all_indents(new_info)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            existing_content = file.read()
    else:
        existing_content = ""

    updated_content = new_info + "\n" + existing_content
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)


def remove_all_indents(text):
    """
    Remove leading whitespace from all lines in text
    
    Args:
        text (str): Input text with potential indentation
        
    Returns:
        str: Text with all leading whitespace removed from each line
    """
    # Split into lines, apply lstrip to each line, then recombine
    lines = text.splitlines()
    stripped_lines = [line.lstrip() for line in lines]
    return "\n".join(stripped_lines)
