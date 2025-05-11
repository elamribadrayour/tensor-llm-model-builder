"""
Main module for the tensorrt-llm-model-builder.
"""

from helpers import builder, env


def run() -> None:
    model_name_unique = builder.get_model_name_unique(model_name=env.model_name)
    paths = builder.get_paths(
        model_name=env.model_name, model_name_unique=model_name_unique
    ).unwrap()
    builder.set_directories(paths=paths).unwrap()
    builder.set_environment(paths=paths, model_name=env.model_name).unwrap()
    builder.get_model_from_hf(model_name=env.model_name).unwrap()

    builder.set_checkpoint(paths=paths).unwrap()
    builder.set_engine(paths=paths).unwrap()
    builder.set_configs(paths=paths).unwrap()

    gcs_paths = builder.get_gcs_paths(model_name_unique=model_name_unique).unwrap()
    builder.save_output(
        paths=paths, gcs_paths=gcs_paths, bucket_name=env.bucket_name
    ).unwrap()


if __name__ == "__main__":
    run()
