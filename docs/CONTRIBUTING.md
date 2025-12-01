# Contributing to ModeLine

First off, thanks for taking the time to contribute! 

The following is a set of guidelines for contributing to ModeLine. These are just guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Quick Start

1.  **Fork and Clone**: Fork the repository to your GitHub account and clone it locally.
    ```bash
    git clone [https://github.com/YOUR-USERNAME/ModeLine.git](https://github.com/YOUR-USERNAME/ModeLine.git)
    cd ModeLine
    ```
2.  **Environment Setup**: Install dependencies.
    ```bash
    pip install -r requirements.txt
    pip install autopep8 pytest  # Dev tools
    ```

## Workflow & Branches

**We do not allow direct commits to `main`.** All changes must come through a Pull Request (PR).

1.  **Create a Branch**: Create a new branch for your specific task.
    * Use the convention: `type/short-description`
    * Examples: `feat/add-login`, `fix/broken-chart`, `docs/update-readme`.
2.  **Make Changes**: Write your code and tests.
3.  **Local Testing**: Before pushing, ensure your code is clean and works.
    ```bash
    # 1. Format Check
    autopep8 --diff --exit-code --recursive .
    # 2. Run Tests
    pytest
    ```
4.  **Commit**: Please follow [Conventional Commits](https://www.conventionalcommits.org/).
    * `feat: add new sidebar filter`
    * `fix: resolve crash on startup`
5.  **Push & PR**: Push your branch and open a Pull Request against `main`.

## Continuous Integration (CI)

Our GitHub Actions workflow enforces high standards. Your PR will be **blocked** if:
* **Linter Fails**: Code is not formatted according to PEP 8 (checked via `autopep8`).
* **Tests Fail**: Any logic error detected by `pytest`.

> **Tip:** Run the checks locally (step 3 above) to avoid waiting for the CI to fail.

## Documentation

* If you change logic, update the documentation/comments accordingly.
* If you add a new dependency, remember to update `requirements.txt`.