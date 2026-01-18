"""
UI Theme and styling for SMF terminal interface.
"""

from typing import Optional


# Color scheme for Rich markup
THEME = {
    # Menu elements
    "option_number": "bold yellow",
    "option_title": "bold cyan",
    "option_desc": "dim",

    # AI-related
    "ai_title": "bold magenta",
    "ai_highlight": "magenta",

    # Status
    "success": "bold green",
    "warning": "yellow",
    "error": "bold red",
    "info": "dim",

    # Data display
    "metric_name": "cyan",
    "metric_value": "green",
    "alpha_value": "yellow",

    # Headers
    "header": "bold cyan",
    "subheader": "blue",
}


def format_menu_item(num: str, title: str, desc: str = "") -> str:
    """
    Format a menu item with consistent styling.

    Args:
        num: Option number/key (e.g., "1", "q")
        title: Main option title
        desc: Optional description

    Returns:
        Rich-formatted string
    """
    parts = [
        f"[{THEME['option_number']}][{num}][/{THEME['option_number']}]",
        f"[{THEME['option_title']}]{title}[/{THEME['option_title']}]",
    ]
    if desc:
        parts.append(f"[{THEME['option_desc']}]{desc}[/{THEME['option_desc']}]")

    return " ".join(parts)


def format_ai_menu_item(num: str, title: str, desc: str = "") -> str:
    """Format an AI-powered menu item with special styling."""
    parts = [
        f"[{THEME['option_number']}][{num}][/{THEME['option_number']}]",
        f"[{THEME['ai_title']}]{title}[/{THEME['ai_title']}]",
    ]
    if desc:
        parts.append(f"[{THEME['option_desc']}]{desc}[/{THEME['option_desc']}]")

    return " ".join(parts)


def styled(text: str, style: str) -> str:
    """Apply a theme style to text."""
    if style in THEME:
        return f"[{THEME[style]}]{text}[/{THEME[style]}]"
    return text


def success(text: str) -> str:
    """Format success message."""
    return f"[{THEME['success']}]✓[/{THEME['success']}] {text}"


def warning(text: str) -> str:
    """Format warning message."""
    return f"[{THEME['warning']}]⚠[/{THEME['warning']}] {text}"


def error(text: str) -> str:
    """Format error message."""
    return f"[{THEME['error']}]✗[/{THEME['error']}] {text}"


def info(text: str) -> str:
    """Format info message."""
    return f"[{THEME['info']}]{text}[/{THEME['info']}]"
