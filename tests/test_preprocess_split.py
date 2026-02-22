from pathlib import Path

def test_split_folders_exist():
    base = Path("data/processed/splits")
    assert (base / "train").exists()
    assert (base / "val").exists()
    assert (base / "test").exists()

def test_train_has_classes():
    train_dir = Path("data/processed/splits/train")
    assert (train_dir / "Cat").exists()
    assert (train_dir / "Dog").exists()