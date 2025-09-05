from huggingface_hub import snapshot_download

snapshot_download(repo_id="bert-base-uncased", cache_dir="./models")
