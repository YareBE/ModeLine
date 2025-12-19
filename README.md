# ModeLine

![License](https://img.shields.io/badge/License-MIT-blue.svg)

**ModeLine** is a web application for creating and visualizing simple and multiple linear regression models with minimal effort.

---

## Contents

- [About](#about)
- [Features](#features)
- [Visuals](#visuals)
- [Installation](#installation)
- [Usage](#usage)
- [Feedback and Contributions](#feedback-and-contributions)
- [License](#license)
- [Contacts](#contacts)

---

## About

ModeLine is an easy-to-use application for creating and visualizing linear regression models from any dataset. Users can upload datasets in multiple formats: .csv, .xslx, .db, etc. But also load previously saved scikit-learn models in a Modeline-Joblib format.

The results can be visualized whit a clean, modern UI. The app provides: the model formula, evaluation metrics (R², MSE), descriptive and interactive charts using Plotly, functionalities to make predictions with your model and the option to download the models you create for reusability.

ModeLine adheres to high standards of flexibility, reusability, and reliability, utilizing a well-known software design methodology (Scrum) and patterns. These patterns ensure the following benefits:

  - Modularity: Different parts of the library can function independently, enhancing the library's modularity and allowing for easier maintenance and updates.
  - Testability: Improved separation of concerns makes the code more testable.
  - Maintainability: Clear structure and separation facilitate better management of the codebase.

This project is ideal for students learning linear regression, beginner programmers, or experienced users like data scientists who need to quickly create regression models.

---

## Features

- Simple and modern web interface
- Multiple dataset format support
- Data preprocessing (null handling, parameters selection, generation seed...)
- Create simple and multiple linear regression models
- Interactive visualization of results
- Save and reload models easily
- Quick predictions from trained models

---

## Installation

Follow this steps to start using ModeLine in your local Machine.
The instructions are separated for Windows and Linux/macOS.

---

### Prerequisites

- Python 3.9+
- Git
- pip (usually included with Python)
- (Optional) Virtual environment

---

### Windows
```bash
# Clone the repository
git clone https://github.com/YareBE/ModeLine.git
cd ModeLine

# Create a virtual environment and activate it (optional but recommended)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/frontend/modeline.py
```

---

### Linux/macOS
```bash
# Clone the repository
git clone https://github.com/YareBE/ModeLine.git
cd ModeLine

# Create a virtual environment and activate it (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/frontend/modeline.py
```

---

## User handbook

Although the app provides an usage guide, here is a well detailed and visual-supported one, to make things as clear as possible:

1. First of all, upload a dataset (`.csv`, `.xlsx`, `.db`...) or an existing model (`.joblib`). You can always upload a new file using this widget.
Please keep in mind that if your dataset has less than 10 rows, some app functionalities described below such as set-splitting will vary or not be able.
![upload screen](./images/upload_screen.png)
2. If you load a modeline-joblib file, you can immediately visualize its results and make predictions.
![loaded model screen](./images/loaded_model_screen.png)
3. On the other hand, if it was a dataset, select the features and target variables for your regression model. 
The dataframe will highlight the selected parameters, you can choose the window of rows to be displayed.
![loaded dataset screen](./images/loaded_dataset_screen.png)
4. Manage any missing values by choosing a substitution method: Mean, Median, Delete, or Constant. The dataframe will show the updated values.
If you change your mind, you can always start from the beginning selecting new parameters and the NA substitution will be undone.
![na handling screen](./images/na_handling_screen.png)
5. Choose a generation seed, training set size and train your model whenever you are ready!
![split train screen](./images/split_train_screen.png)
6. Visualize the plots and metrics and make predictions with your model
![predictions screen](./images/predictions_screen.png)
7. Choose a name, a description, and store your model locally
![saving model screen](./images/saving_model_screen.png)

---

## Feedback and Contributions
We've made every effort to make out application the most complete possible, however we may have encountered many errors. Whether you have feedback on features, have encountered any bugs, or have suggestions for enhancements, we're eager to hear from you. Your insights make ModeLine more robust and user-friendly.

Please feel free to contribute by submitting an issue or joining the discussions. Each contribution helps us to grow and improve. Please check [CONTRIBUTING.md](https://github.com/YareBE/ModeLine/blob/main/CONTRIBUTING.md) for further details.

We appreciate your support and look forward to making our product even better with your help!

---

## License

This software is licensed under the [MIT LICENSE](https://github.com/YareBE/ModeLine/blob/main/LICENSE)

---

## Contacts

For more details about out products, services, or any general information regarding the application, feel free to reach out to us. We are here to provide support and answer any questions you may have. You can contact our team at:

- Pablo Fernández Ríos — [pablo.fernandez.rios@udc.es](mailto:pablo.fernandez.rios@udc.es)
- Yare Brea Espinosa — [yare.bespinosa@udc.es](mailto:yare.bespinosa@udc.es)
- Rodrigo Marino Álvarez — [rodrigo.marino.alvarez@udc.es](mailto:rodrigo.marino.alvarez@udc.es)

We look forward to assisting you and ensuring your experience with our product is succesful and enjoyable!

