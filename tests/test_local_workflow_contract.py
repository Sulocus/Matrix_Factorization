from pathlib import Path


FORBIDDEN_WORKFLOW_TERMS = [
    "CODEX" + "_WEB",
    "Codex " + "web",
    "Codex " + "Web",
    "clo" + "ud-safe",
    "smo" + "ketest",
    "smo" + "ke test",
    "tests/" + "smo" + "ke",
    "云" + "端",
]


def test_no_obsolete_remote_or_legacy_quick_workflow_terms_remain():
    roots = [Path("README.md"), Path("AGENTS.md"), Path("docs"), Path("src"), Path("tests"), Path("configs"), Path("trials")]
    offenders = []
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.resolve() == Path(__file__).resolve():
                continue
            if not path.is_file() or path.suffix not in {".md", ".py", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for term in FORBIDDEN_WORKFLOW_TERMS:
                if term in text:
                    offenders.append(f"{path}:{term}")

    assert offenders == []
