import json
import os
import pickle
from typing import Any, Dict


def get_project_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".nlp_projects")


def save_obj(path: str, data: Any) -> None:
    with open(path, "wb") as fp:
        pickle.dump(data, fp)


def load_obj(path: str) -> Any:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def save_json(path: str, data: Dict[Any, Any]) -> None:
    with open(path, "w") as fp:
        json.dump(data, fp)


def load_json(path: str) -> Dict[Any, Any]:
    with open(path, "r") as fp:
        return json.load(fp)
