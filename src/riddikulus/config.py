"""Portable access to the frozen H4 ontology and controller configuration."""
from importlib.resources import files
import json

def default_controller_config():
    return json.loads(files("riddikulus").joinpath("data/controller_config.json").read_text())

def default_hierarchy_path():
    return str(files("riddikulus").joinpath("data/hierarchy_h4_atomic.json"))
