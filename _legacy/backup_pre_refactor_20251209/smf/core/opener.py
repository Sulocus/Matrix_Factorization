"""
Cross-platform file opener.

Opens files with the system's default application.
Supports WSL (Windows Subsystem for Linux).
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
from typing import Union


def _is_wsl() -> bool:
    """Check if running in WSL (Windows Subsystem for Linux)."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


def _wsl_path_to_windows(path: Path) -> str:
    """Convert WSL path to Windows path for explorer.exe."""
    # Use wslpath to convert
    try:
        result = subprocess.run(
            ['wslpath', '-w', str(path.resolve())],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    # Fallback: manual conversion for /home/user -> \\wsl$\...
    return str(path.resolve())


def open_file(file_path: Union[str, Path]) -> bool:
    """
    Open a file with the system's default application.

    Works on:
    - Windows: uses os.startfile or start command
    - macOS: uses open command
    - Linux: uses xdg-open
    - WSL: uses explorer.exe or wslview

    Args:
        file_path: Path to the file to open

    Returns:
        True if successfully opened, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False

    try:
        if sys.platform == 'win32':
            # Windows
            os.startfile(str(file_path))
        elif sys.platform == 'darwin':
            # macOS
            subprocess.run(['open', str(file_path)], check=True)
        elif _is_wsl():
            # WSL: use Windows explorer or wslview
            win_path = _wsl_path_to_windows(file_path)
            # Try wslview first (from wslu package)
            try:
                subprocess.run(['wslview', str(file_path)], check=True,
                             stderr=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.CalledProcessError):
                # Fallback to explorer.exe
                subprocess.run(['explorer.exe', win_path], check=False)
        else:
            # Linux and other Unix-like systems
            subprocess.run(['xdg-open', str(file_path)], check=True)
        return True
    except Exception as e:
        print(f"Error opening file: {e}")
        return False


def open_image(image_path: Union[str, Path]) -> bool:
    """
    Open an image file with the system's default image viewer.

    Args:
        image_path: Path to the image file

    Returns:
        True if successfully opened
    """
    return open_file(image_path)


def open_folder(folder_path: Union[str, Path]) -> bool:
    """
    Open a folder in the system's file explorer.

    Args:
        folder_path: Path to the folder

    Returns:
        True if successfully opened
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return False

    try:
        if sys.platform == 'win32':
            os.startfile(str(folder_path))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(folder_path)], check=True)
        elif _is_wsl():
            # WSL: use Windows explorer
            win_path = _wsl_path_to_windows(folder_path)
            subprocess.run(['explorer.exe', win_path], check=False)
        else:
            subprocess.run(['xdg-open', str(folder_path)], check=True)
        return True
    except Exception as e:
        print(f"Error opening folder: {e}")
        return False
