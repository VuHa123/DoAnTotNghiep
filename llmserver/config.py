def load_url(filepath="API_LLM_SERVER.txt") -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        url = f.readline().strip()
    return url + "/model/generate/"

API_LLM = load_url()
