"""
Preset manager for saving and loading experiment configurations.

Presets are stored as YAML files in:
- Built-in: smf/presets/builtin/
- User: ~/.smf/presets/

User presets take precedence over built-in presets.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


# Directory paths
BUILTIN_DIR = Path(__file__).parent / "builtin"
USER_DIR = Path.home() / ".smf" / "presets"


class PresetManager:
    """
    Manages experiment presets.

    Usage:
        manager = PresetManager()

        # List available presets
        presets = manager.list_presets()

        # Load a preset
        config = manager.load_preset("quick_test")

        # Save current config as preset
        manager.save_preset("my_config", config)
    """

    def __init__(self):
        self.builtin_dir = BUILTIN_DIR
        self.user_dir = USER_DIR

    def list_presets(self) -> List[Dict[str, Any]]:
        """
        List all available presets.

        Returns:
            List of preset info dictionaries with 'name', 'source', 'description'
        """
        presets = []

        # Built-in presets
        if self.builtin_dir.exists():
            for path in sorted(self.builtin_dir.glob("*.yaml")):
                info = self._get_preset_info(path, source="builtin")
                if info:
                    presets.append(info)

        # User presets (can override built-in)
        if self.user_dir.exists():
            for path in sorted(self.user_dir.glob("*.yaml")):
                info = self._get_preset_info(path, source="user")
                if info:
                    # Check if overriding built-in
                    existing = next((p for p in presets if p['name'] == info['name']), None)
                    if existing:
                        presets.remove(existing)
                        info['overrides'] = 'builtin'
                    presets.append(info)

        return presets

    def load_preset(self, name: str) -> Dict[str, Any]:
        """
        Load a preset by name.

        Args:
            name: Preset name (without .yaml extension)

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If preset not found
        """
        # Try user directory first (higher priority)
        user_path = self.user_dir / f"{name}.yaml"
        if user_path.exists():
            return self._load_yaml(user_path)

        # Try built-in directory
        builtin_path = self.builtin_dir / f"{name}.yaml"
        if builtin_path.exists():
            return self._load_yaml(builtin_path)

        raise ValueError(f"Preset '{name}' not found")

    def save_preset(self, name: str, config: Dict[str, Any], description: str = "") -> Path:
        """
        Save a preset to user directory.

        Args:
            name: Preset name
            config: Configuration dictionary
            description: Optional description

        Returns:
            Path to saved preset
        """
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata
        preset_data = {
            '_description': description,
            **config,
        }

        path = self.user_dir / f"{name}.yaml"
        with open(path, 'w') as f:
            yaml.dump(preset_data, f, default_flow_style=False, allow_unicode=True)

        return path

    def delete_preset(self, name: str) -> bool:
        """
        Delete a user preset.

        Args:
            name: Preset name

        Returns:
            True if deleted, False if not found or is built-in
        """
        user_path = self.user_dir / f"{name}.yaml"
        if user_path.exists():
            user_path.unlink()
            return True
        return False

    def _get_preset_info(self, path: Path, source: str) -> Optional[Dict[str, Any]]:
        """Get basic info about a preset without loading full config."""
        try:
            with open(path, 'r') as f:
                # Only read first few lines for description
                content = f.read(1000)
                data = yaml.safe_load(content)

            return {
                'name': path.stem,
                'source': source,
                'description': data.get('_description', ''),
                'path': str(path),
            }
        except Exception:
            return None

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Remove metadata
        if '_description' in data:
            del data['_description']

        return data


# Convenience functions
_manager = None


def _get_manager() -> PresetManager:
    global _manager
    if _manager is None:
        _manager = PresetManager()
    return _manager


def list_presets() -> List[Dict[str, Any]]:
    """List all available presets."""
    return _get_manager().list_presets()


def load_preset(name: str) -> Dict[str, Any]:
    """Load a preset by name."""
    return _get_manager().load_preset(name)


def save_preset(name: str, config: Dict[str, Any], description: str = "") -> Path:
    """Save a preset."""
    return _get_manager().save_preset(name, config, description)
