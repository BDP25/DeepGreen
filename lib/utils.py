def load_eval_script(path: str) -> str:
    with open(path, "r") as f:
        eval_script = f.read()
    return eval_script