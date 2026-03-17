# Task: Add --version flag to rag_cli.py

rag_cli.py has no version flag. Add one so users can quickly confirm which version they have.

## What to add

Add `--version` / `-V` to the argparse top-level parser in rag_cli.py.
Print a single line: `rag-pipeline 1.0.0`.

Also add the usage line to the module docstring:
```
    python rag_cli.py --version
```

## Constraints

- Only touch `rag_cli.py`
- Do not change any logic, just the parser setup and docstring
- Keep changes minimal

## Validation

Run from the repo root directory:
```
python rag_cli.py --version
grep "\-\-version" rag_cli.py
```

Both should succeed. The first should print `rag-pipeline 1.0.0` and exit 0.
The second should show the version argument line.
