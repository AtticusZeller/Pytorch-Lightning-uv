from pathlib import Path

import pytest

from src.python_uv.advanced import Animal, Dog, fibonacci, use_context_manager


def test_fibonacci() -> None:
    fib = list(fibonacci(5))
    assert fib == [0, 1, 1, 2, 3]


def test_animal() -> None:
    with pytest.raises(NotImplementedError):
        Animal("Generic").speak()


def test_dog() -> None:
    dog = Dog("Buddy")
    assert dog.speak() == "Buddy says Woof!"


def test_context_manager(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!")
    assert use_context_manager(str(test_file)) == "Hello, World!"
