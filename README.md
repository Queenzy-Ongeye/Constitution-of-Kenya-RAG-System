# Kenyan Constitution RAG System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system specifically designed for the Kenyan Constitution. The system combines advanced language models with constitutional document retrieval to provide accurate and contextually relevant responses about Kenya's constitutional matters.

## Features
- Document-based retrieval system for the Kenyan Constitution
- Natural language query processing
- Context-aware response generation
- Interactive interface through Jupyter notebooks

## Project Structure
```
.
├── index.ipynb          # Main Jupyter notebook interface
├── ken127322.pdf        # Kenyan Constitution document
└── env/                 # Virtual environment for dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Virtual environment (recommended)

### Installation
1. Clone the repository
2. Create and activate virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Activate the virtual environment
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `index.ipynb`
4. Follow the instructions in the notebook to interact with the RAG system

## System Architecture
The RAG system consists of the following components:

1. **Document Processing**
   - PDF parsing and text extraction
   - Document chunking and embedding
   - Vector database storage

2. **Query Processing**
   - Natural language understanding
   - Query embedding
   - Similarity search

3. **Response Generation**
   - Context retrieval
   - Language model integration
   - Response formulation

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify License]

## Contact
[Add contact information] 