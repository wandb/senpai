import tomllib
from fnmatch import fnmatchcase
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_setuptools_discovers_only_source_packages():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    find = config["tool"]["setuptools"]["packages"]["find"]
    packages = {
        ".".join(path.parent.relative_to(ROOT).parts)
        for path in (ROOT / "senpai_agent").rglob("__init__.py")
    }

    assert find == {"include": ["senpai_agent*"], "namespaces": False}
    assert packages
    assert all(
        fnmatchcase(package, find["include"][0]) for package in packages
    )


def test_python_prompt_module_requires_no_package_data():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert "package-data" not in config["tool"]["setuptools"]
    assert (ROOT / "senpai_agent" / "PROMPTS.py").is_file()
    assert not (ROOT / "senpai_agent" / "PROMPTS.md").exists()
