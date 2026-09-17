#!/bin/zsh
set -e

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/streamlit" ]; then
  echo "Streamlit was not found in .venv."
  echo "Please run: .venv/bin/pip install -r requirements.txt"
  read "?Press Enter to close..."
  exit 1
fi

echo "Starting Local RAG Assistant Demo..."
echo "If the browser does not open automatically, go to: http://localhost:8501"
echo "Press Ctrl+C in this window to stop the demo."
echo ""

.venv/bin/streamlit run demo_app.py
