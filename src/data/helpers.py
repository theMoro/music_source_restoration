"""
Helper functions for dataset utilities.
"""

import os
from pathlib import Path
from typing import Optional, Union


def get_absolute_path(
    relative_path: str, prefix_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Convert a relative path from the dataset to an absolute path.
    
    If prefix_path is set, joins it with the relative path.
    If the path is already absolute, returns it as-is.
    
    Args:
        relative_path: Relative path from the HuggingFace dataset
        prefix_path: Optional prefix path to join with relative_path
        
    Returns:
        str: Absolute path to the audio file
    """
    if not relative_path:
        return relative_path
    
    # If path is already absolute, return as-is
    if os.path.isabs(relative_path):
        return relative_path
    
    # If prefix_path is set, use it to construct absolute path
    if prefix_path is not None:
        return str(Path(prefix_path) / relative_path)
    
    # Fallback: assume path is already absolute or relative to current directory
    return relative_path

