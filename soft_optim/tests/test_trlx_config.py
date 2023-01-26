from soft_optim.trlx_config import default_config_override


class TestDefaultConfigOverride:
    def test_default_config_works(self):
        config = default_config_override({})
        assert config.method.name == "ppoconfig"

    def test_sets_float_val(self):
        config = default_config_override({
            "method.init_kl_coef": 0.01
        })
        assert config.method.init_kl_coef == 0.01  # type: ignore

    def test_sets_int_val(self):
        config = default_config_override({
            "train.batch_size": 1.01
        })
        assert config.train.batch_size == 1
