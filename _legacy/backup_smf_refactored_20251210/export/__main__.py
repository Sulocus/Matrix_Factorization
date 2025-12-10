"""
Entry point for: python -m smf.export

Usage:
    python -m smf.export          # Generate all scripts
    python -m smf.export.bundle   # Same as above
"""

from .bundler import bundle_all

if __name__ == "__main__":
    scripts = bundle_all()
    print(f"\nGenerated {len(scripts)} scripts:")
    for script in scripts:
        print(f"  - {script}")
