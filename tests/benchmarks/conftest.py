"""pytest configuration for benchmark tests.

Registers the ``benchmark`` mark so it does not produce PytestUnknownMarkWarning
and can be used to selectively run slow ground-truth validation tests:

    pytest tests/benchmarks/ -v -m benchmark
"""


def pytest_configure(config: object) -> None:
    config.addinivalue_line(  # type: ignore[union-attr]
        "markers",
        "benchmark: marks slow ground-truth validation tests that exercise the"
        " full analysis pipeline (run with: pytest tests/benchmarks/ -v -m benchmark)",
    )
