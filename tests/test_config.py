from aircraft_damage.config import load_config
from aircraft_damage.utils import project_root


def test_inference_config_loads_and_resolves_paths() -> None:
    config = load_config("configs/inference.yaml")

    assert config["project"]["name"] == "aircraft_damage_vgg16_blip"
    assert config["paths"]["results_dir"] == str(project_root() / "results")
    assert config["classifier"]["class_names"] == ["crack", "dent"]
