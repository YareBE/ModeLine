# Contributing to ModeLine

First off, thanks for taking the time to contribute! 

The following is a set of guidelines for contributing to ModeLine. These are just guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Quick Start

If you have not already done it, set up the repository on your local machine by following the installation guide:

[Installation Guide](https://github.com/YareBE/ModeLine/blob/main/README.md#installation)

## Project Structure

Here is the current repository layout. Please ensure you place new files in the correct directories:

```text
ModeLine/
├── .github/
│   └── workflows/
│       └── ci.yaml
├── images/                   # Visual resources
│   ├── loaded_dataset_screen.png
│   ├── loaded_model_screen.png
│   ├── na_handling_screen.png
│   ├── predictions_screen.png
│   ├── saving_model_screen.png
│   ├── split_train_screen.png
│   └── upload_screen.png       
├── src/
│   ├── backend/              # Core logic (Training, Serialization)
│   │   ├── uploader.py
│   │   ├── trainer.py
│   │   ├── preprocessing.py
│   │   └── serializer.py
│   └── frontend/             # Streamlit UI components
│       ├── display_utils.py
│       ├── modeline.py
│       ├── preprocessing_gui.py
│       ├── serializer_gui.py
│       ├── trainer_gui.py
│       └── uploader_gui.py
├── tests/                    # Automated unit tests
│   ├── test_preprocessing.py
│   ├── test_serializer.py
│   ├── test_trainer.py
│   ├── test_uploader.py
│   └── TESTS_PLAN.md          # Functional testing guide
├── .gitignore
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE
├── README.md
├── requirements.txt          # Production dependencies
└── requirements-dev.txt      # Development dependencies
```

## Workflow & Branches

**We do not allow direct commits to `main`.** All changes must come through a Pull Request (PR).

1.  **Create a Branch**: Create a new branch for your specific task.
    * Use the convention: `type/short-description`
    * Examples: `feat/add-login`, `fix/broken-chart`, `docs/update-readme`.
2.  **Make Changes**: Write your code and tests.
3.  **Local Testing**: Before pushing, ensure your code is clean and works.
    ```bash
    # 1. Install Development Dependencies
    pip install -r requirements-dev.txt
    # 2. Format Check
    autopep8 --diff --exit-code --recursive ./src
    # 3. Run Tests
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
