# semantic-analysis

# To-do
    - Text Embeddings
    - Translation Capabilites
    - Topic Modelling
    - Clustering
    - Semantic Searching


# Setup
python3 -m venv venv
source venv/bin/activate
pip install pip-tools
pip install setuptools
pip-sync build-requirements.txt requirements.txt

# Rebuilding Requirements
pip-compile requirements.in --no-emit-index-url
pip-compile build-requirements.in --no-emit-index-url

