import importlib.util
import sys

from omegaconf import OmegaConf


def load_config_py(path):
    spec = importlib.util.spec_from_file_location("config", path)

    try:
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except FileNotFoundError:
        raise FileNotFoundError(f"No python config file ({path})")

    return config


def get_omegaconf_config(dataclass_config,
                         from_yamls=None,
                         from_cli_yaml=True,
                         from_cli=True):
    config = OmegaConf.structured(dataclass_config)
    if len(sys.argv[1:]) and sys.argv[1] in ("-h", "--help"):
        print(OmegaConf.to_yaml(config))
        if from_cli_yaml:
            print("Use --config=<path> to load yaml config, can be used multiple times, \n"
                  "can be combined with omegaconf dotlist arguments")
        sys.exit(0)

    try:
        if from_yamls is not None:
            for yaml_file in from_yamls:
                yaml_config = OmegaConf.load(yaml_file)
                config = OmegaConf.merge(config, yaml_config)

        if from_cli_yaml:
            from_yamls = [arg[9:] for arg in sys.argv[1:] if arg.startswith("--config=")]
            for yaml_file in from_yamls:
                yaml_config = OmegaConf.load(yaml_file)
                config = OmegaConf.merge(config, yaml_config)

        if from_cli:
            arguments = sys.argv[1:]  # skip program name
            if from_cli_yaml:
                arguments = [arg for arg in arguments if not arg.startswith("-")]
            cli = OmegaConf.from_cli(arguments)
            config = OmegaConf.merge(config, cli)
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        print(OmegaConf.to_yaml(dataclass_config))
        raise e

    return config
