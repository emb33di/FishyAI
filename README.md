# Property Law AI Assistant

An AI-powered assistant for Property Law exam preparation that uses your PDF materials as context for answering questions.

## Features

- Load and process Property Law PDFs
- Ask questions about Property Law concepts
- Get answers based on your provided materials
- Maintains conversation history for context-aware responses

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/property-law-ai.git
cd property-law-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. Place your Property Law PDFs in the `pdfs` directory

## Usage

Run the application:
```bash
python app.py
```

Type your questions about Property Law, and the AI will provide answers based on your PDF materials.

## Project Structure

- `agent.py`: Core AI agent implementation
- `app.py`: Command-line interface
- `pdfs/`: Directory for your Property Law PDFs
- `requirements.txt`: Project dependencies

## Note

Make sure to keep your API keys secure and never commit them to the repository. 