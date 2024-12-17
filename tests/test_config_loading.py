def test_config_loading():
    from gtp.configs.loaders import load_configs

    try:
        configs = load_configs("configs/default.yaml")
    except:
        assert False, "Assertion Error in loading configs"

    