def validate_stage1_config(config) -> None:
    if not config.stage_1.target:
        raise ValueError("Config must provide a 'stage_1' section with target.")
    if not config.gan.loss:
        raise ValueError("Config must define a top-level 'gan' section.")
    if config.dataset.type not in {"hf", "imagefolder", "concat", "fluffyelephant", "multiview"}:
        raise ValueError(
            f"dataset.type must be 'hf', 'imagefolder', 'concat', 'fluffyelephant', or 'multiview', "
            f"got '{config.dataset.type}'"
        )
