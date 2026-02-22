from pathlib import Path

def test_project_structure_exists():
    assert Path("src").exists()
    assert Path("app").exists()
    assert Path("tests").exists()

    #test