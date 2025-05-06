<div align="left" style="position: relative;">
<!-- <img src="https://img.icons8.com/?size=512&id=55494&format=png" align="right" width="30%" style="margin: -20px 0 0 20px;"> -->
<h1>FALCON</h1>
<p align="left">
	<em>Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning.</em>
</p>
<p align="left">
	<img src="https://img.shields.io/github/license/ngocminhta/FALCON?style=flat-square&logo=opensourceinitiative&logoColor=white&color=00a3b9" alt="license">
	<img src="https://img.shields.io/github/last-commit/ngocminhta/FALCON?style=flat-square&logo=git&logoColor=white&color=00a3b9" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/ngocminhta/FALCON?style=flat-square&color=00a3b9" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/ngocminhta/FALCON?style=flat-square&color=00a3b9" alt="repo-language-count">
</p>
<p align="left">Built with the tools and technologies:</p>
<p align="left">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat-square&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
</p>
</div>
<br clear="right">

## 🔗 Table of Contents

- [🔗 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#️-prerequisites)
  - [⚙️ Installation](#️-installation)
  - [🤖 Usage](#-usage)
  - [🧪 Testing](#-testing)
- [📌 News](#-news)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

The FALCON project revolutionizes the detection of deepfake content through advanced text analysis. By leveraging state-of-the-art machine learning techniques, it offers robust tools for generating, managing, and evaluating text embeddings to accurately classify content as human, AI-generated, or mixed. Ideal for tech companies and cybersecurity experts, FALCON enhances digital trust and integrity across various media platforms.

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes a modular approach with separate scripts for training, inference, and database management.</li><li>Incorporates distributed processing capabilities to enhance scalability.</li><li>Employs FAISS for efficient indexing and vector search operations.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Adheres to modern Python practices with clear modular separation in scripts and utility functions.</li><li>Includes comprehensive script files for training, testing, and inference to maintain operational clarity and separation.</li><li>Uses advanced machine learning libraries and frameworks for robust algorithm development.</li></ul> |
| 📄 | **Documentation** | <ul><li>Documentation includes detailed command guides for installation, usage, and testing.</li><li>Code comments and structured documentation are present in critical scripts such as `train_classifier.py` and `infer.py`.</li><li>Utilizes Markdown badges and links for easy navigation and reference in documentation.</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with Python's `pip` for dependency management as seen in `requirements.txt`.</li><li>Supports various machine learning and data processing libraries.</li><li>Capable of utilizing both CPU and GPU resources for processing.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Highly modular design with separate components for different functionalities like training, inference, and database management.</li><li>Scripts such as `train.sh` and `infer.sh` enhance modularity by encapsulating functionality.</li><li>Easy to extend or modify individual components without affecting others.</li></ul> |
| 🧪 | **Testing**       | <ul><li>Includes specific scripts for testing like `test_from_database.sh`.</li><li>Utilizes `pytest` for running tests, ensuring code reliability and functionality.</li><li>Testing scripts are adaptable to various datasets and configurations.</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimized for high performance with support for distributed computing and GPU acceleration.</li><li>FAISS integration for efficient large-scale vector handling and searches.</li><li>Performance metrics such as accuracy, precision, recall, and F1 score are meticulously tracked and optimized.</li></ul> |
| 🛡️ | **Security**      | <ul><li>Focuses on robust data handling and processing to ensure integrity of machine learning models.</li><li>Implements checks and balances in data serialization and deserialization processes.</li><li>Secure handling of text data to prevent data leakage or misuse.</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Managed through `pip` and a well-defined `requirements.txt` file.</li><li>Dependencies include essential libraries for neural network modeling, natural language processing, and data visualization.</li><li>Ensures compatibility and functionality across various environments.</li></ul> |

---

## 📁 Project Structure

```sh
└── FALCON/
    ├── README.md
    ├── algorithm
    │   ├── gen_database.py
    │   ├── infer.py
    │   ├── requirements.txt
    │   ├── script
    │   │   ├── gen_database.sh
    │   │   ├── infer.sh
    │   │   ├── test.sh
    │   │   ├── test_from_database.sh
    │   │   └── train.sh
    │   ├── src
    │   │   ├── __init__.py
    │   │   ├── index.py
    │   │   ├── simclr.py
    │   │   └── text_embedding.py
    │   ├── test_from_database.py
    │   ├── train_classifier.py
    │   └── utils
    │       ├── __init__.py
    │       ├── load_dataset.py
    │       └── utils.py
    └── data
        ├── FALCONSet
        │   ├── deepseek-text
        │   ├── gemini-2.0-text
        │   ├── gpt-4o-mini-text
        │   ├── human---deepseek-text
        │   ├── human---gemini-2.0-text
        │   ├── human---gpt-4o-mini-text
        │   ├── human---llama-text
        │   ├── human-text
        │   └── llama-text
        ├── Unseen_Domain
        │   ├── deepseek-text
        │   ├── gemini-2.0-text
        │   ├── gpt-4o-mini-text
        │   ├── human---deepseek-text
        │   ├── human---gemini-2.0-text
        │   ├── human---gpt-4o-mini-text
        │   ├── human---llama-text
        │   ├── human-text
        │   └── llama-text
        ├── Unseen_Domain_and_Unseen_Generator
        │   ├── gemma-text
        │   ├── human---gemma-text
        │   ├── human---mistral-text
        │   ├── human---qwen-text
        │   ├── human-text
        │   ├── mistral-text
        │   └── qwen-text
        └── Unseen_Generator
            ├── gemma-text
            ├── human---gemma-text
            ├── human---mistral-text
            ├── human---qwen-text
            ├── human-text
            ├── mistral-text
            └── qwen-text
```


### 📂 Project Index
<details open>
	<summary><b><code>FALCON/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			</table>
		</blockquote>
	</details>
	<details> <!-- algorithm Submodule -->
		<summary><b>algorithm</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/test_from_database.py'>test_from_database.py</a></b></td>
				<td>- Evaluates the performance of text embedding models on test datasets, both in-domain and out-of-domain, using fuzzy k-nearest neighbors classification<br>- It measures accuracy, precision, recall, F1 score, MSE, and MAE across different values of K, optimizing model parameters and visualizing results to identify the best configuration.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/infer.py'>infer.py</a></b></td>
				<td>- `algorithm/infer.py` serves as the inference module, utilizing a pre-trained text embedding model to process input text, generate embeddings, and perform k-nearest neighbors search against a serialized index<br>- It classifies the text based on the proximity of its embeddings to known labeled data, determining if the source is human, AI, or mixed.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/gen_database.py'>gen_database.py</a></b></td>
				<td>- Generates and manages embeddings for textual data, facilitating the creation and updating of searchable databases<br>- It supports distributed processing for scalability and includes functionality to handle both in-domain and out-of-domain datasets, ensuring robustness across different data types<br>- The script also provides tools for serialization and deserialization of indexed data.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/requirements.txt'>requirements.txt</a></b></td>
				<td>- Specifies the dependencies required for the algorithm component of the project, ensuring compatibility and functionality across various machine learning and data processing libraries<br>- It includes essential libraries for data manipulation, progress tracking, neural network modeling, natural language processing, and visualization, facilitating a robust environment for algorithm development and experimentation.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/train_classifier.py'>train_classifier.py</a></b></td>
				<td>- The `train_classifier.py` file is a critical component of the project's machine learning pipeline, specifically designed for training classification models<br>- This script integrates various machine learning and deep learning libraries and frameworks to set up, train, and evaluate classifiers using textual data<br>- Its primary function is to orchestrate the training process, which includes data loading, model initialization, setting up the training loop, and logging the training progress.

Key functionalities of this script include:
1<br>- **Data Preparation**: It loads and preprocesses text data, ensuring it is in the correct format for training using utilities like `load_dataset` and tokenization processes.
2<br>- **Model Setup**: It initializes models for text classification, leveraging pre-trained models and custom classifier layers, which are crucial for handling the specifics of the text data.
3<br>- **Training Loop**: The script manages the training process, including setting random seeds for reproducibility, configuring data loaders for batch processing of data, and defining the optimization strategy.
4<br>- **Logging and Metrics**: Utilizes `SummaryWriter` for TensorBoard to log training metrics and progress, which is vital for monitoring the training process and outcomes.
5<br>- **Evaluation**: Implements functions to compute and calculate metrics, aiding in the assessment of the classifier's performance during and after training.

In the broader scope of the project, `train_classifier.py` serves as the executable script that directly impacts the model's performance by handling the training operations efficiently<br>- It interacts with various modules of the project such as `src` for model definitions, `utils` for utility functions, and `lightning` for leveraging PyTorch Lightning capabilities, indicating its integral role in the machine learning workflow of the codebase.</td>
			</tr>
			</table>
			<details>
				<summary><b>src</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/src/text_embedding.py'>text_embedding.py</a></b></td>
						<td>- TextEmbeddingModel, defined in algorithm/src/text_embedding.py, serves as a core component for generating normalized text embeddings using pre-trained transformer models<br>- It supports various pooling strategies and can handle different transformer architectures, adapting its behavior based on the model's specific characteristics to optimize text representation within the broader system.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/src/index.py'>index.py</a></b></td>
						<td>- Indexer, a core component within the algorithm module, manages the indexing, searching, and serialization of vector data using FAISS<br>- It supports operations on GPU, handles large-scale vector searches efficiently, and maintains a mapping between internal and external identifiers, facilitating quick retrieval and robust data management across the system's architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/src/simclr.py'>simclr.py</a></b></td>
						<td>- Defines and implements a SimCLR-based classifier for text embeddings, integrating a classification head for sentence-level tasks<br>- It utilizes cosine similarity for contrastive learning, supports distributed computing with fabric, and handles various classification scenarios including cross-entropy loss calculations for different label types, enhancing model training and evaluation flexibility within the project's architecture.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>script</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/script/test_from_database.sh'>test_from_database.sh</a></b></td>
						<td>- Executes testing of deepfake detection models using a specified dataset and model checkpoint<br>- It configures the test environment for different modes including deepfake, TuringBench, and M4 challenges, both monolingual and multilingual, adjusting parameters like device number and batch size to optimize performance.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/script/train.sh'>train.sh</a></b></td>
						<td>- Train.sh initiates the training of a classifier for detecting deepfakes using a pre-trained RoBERTa model from Princeton NLP<br>- It configures the model to process data from a specified path, setting parameters like device number, batch size, learning rate, and epochs, focusing on optimizing performance across different datasets within the broader architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/script/infer.sh'>infer.sh</a></b></td>
						<td>- Executes a deep learning model for detecting deepfake content by leveraging a pre-trained model and a specified database<br>- The script processes a given text input to assess and output the likelihood of the content being a deepfake, using the top 5 predictions<br>- This functionality is crucial for maintaining the integrity and trustworthiness of digital media within the system.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/script/test.sh'>test.sh</a></b></td>
						<td>- Executes a Python script for testing a deepfake detection model using a K-nearest neighbors algorithm<br>- It configures the script to process data from a specified path, utilizing a pre-trained model, and saves the resulting database for further analysis<br>- The script is adaptable for various datasets but is currently set for the deepfake domain.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/script/gen_database.sh'>gen_database.sh</a></b></td>
						<td>- Generates a training database for detecting deepfake content by executing a Python script configured for high-throughput processing on multiple devices<br>- It specifically prepares a dataset using a pre-trained model, targeting the deepfake detection domain, and stores the results in a structured database format for further use in model training and evaluation.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>utils</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/utils/load_dataset.py'>load_dataset.py</a></b></td>
						<td>Manages the loading and preprocessing of datasets for machine learning models, supporting various dataset configurations including "falconset," "llmdetectaive," and "hart." It categorizes text data into human, human+AI, or AI-generated content, enriching each entry with labels and indices for further processing and analysis in model training and evaluation phases.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/ngocminhta/FALCON/blob/master/algorithm/utils/utils.py'>utils.py</a></b></td>
						<td>- Provides utility functions for evaluating and reporting machine learning model performance across various metrics such as accuracy, precision, recall, and F1 score<br>- It includes specialized functions for handling multi-class scenarios, binary outcomes, and specific use cases involving embeddings and unique identifiers, enhancing the interpretability of model results in diverse contexts.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with FALCON, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


### ⚙️ Installation

Install FALCON using one of the following methods:

**Build from source:**

1. Clone the FALCON repository:
```sh
❯ git clone https://github.com/ngocminhta/FALCON
```

2. Navigate to the project directory:
```sh
❯ cd FALCON
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r algorithm/requirements.txt
```




### 🤖 Usage
Run FALCON using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

To train the model
```sh
❯ python algorithm/train_classifier.py <your parameter goes here>
```
To generate the vector database after training
```sh
❯ python algorithm/gen_database.py <your parameter goes here>
```

### 🧪 Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python algorithm/test_from_database.py <your parameter goes here>
```


---
## 📌 News

**[2025.05.06]** Our project is publicly accessible.

---

## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/ngocminhta/FALCON/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/ngocminhta/FALCON/issues)**: Submit bugs found or log feature requests for the `FALCON` project.
- **💡 [Submit Pull Requests](https://github.com/ngocminhta/FALCON/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/ngocminhta/FALCON
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/ngocminhta/FALCON/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=ngocminhta/FALCON">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the [MIT](LICENSE.md) License.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---