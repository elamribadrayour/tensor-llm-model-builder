"""
Builder for the ModelBuilder class.
"""

import os
import subprocess
from uuid import uuid4
from pathlib import Path

from loguru import logger
from result import Err, Ok, Result
from huggingface_hub import hf_hub_download

from helpers import save


def run_command(command: str, cwd: str | None = None) -> Result[bool, str]:
    """Execute a shell command and handle errors."""
    try:
        logger.info(f"Executing command: {command}")
        result = subprocess.run(
            command, shell=True, check=True, cwd=cwd, capture_output=True, text=True
        )
        logger.info(result.stdout)
        return Ok(True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        return Err(e.stderr)


def get_model_name_unique(model_name: str) -> str:
    """Get the unique model name."""
    return f"{model_name}-{uuid4()}"


def get_paths(model_name: str, model_name_unique: str) -> Result[dict[str, Path], str]:
    """Get the paths for the model."""
    base_path = Path("/data")
    if not base_path.exists():
        raise Exception(
            f"Base path {base_path} does not exist, a pvc of at least 1Ti mounted on /data is required because of large file sizes"
        )

    config_path = hf_hub_download("meta-llama/" + model_name, filename="config.json")
    hf_model_path = Path(config_path).parent

    ckpt_path = base_path / "tensorrt" / "ckpt" / model_name_unique
    engine_path = base_path / "tensorrt" / "engines" / model_name_unique
    backend_path = Path("/tmp/tensorrtllm_backend")

    if not backend_path.exists():
        raise Exception(
            f"Backend path {backend_path} does not exist, a pvc of at least 1Ti mounted on /data is required because of large file sizes"
        )

    logger.info(f"Paths: {ckpt_path}, {engine_path}, {backend_path}, {hf_model_path}")

    return Ok(
        {
            "base": base_path,
            "ckpt": ckpt_path,
            "engine": engine_path,
            "backend": backend_path,
            "hf_model": hf_model_path,
        }
    )


def get_gcs_paths(model_name_unique: str) -> Result[dict[str, str], str]:
    """Get the GCS paths for the model."""
    prefix_gcs_path = "tensorrt-llm/models"

    return Ok(
        {
            "ckpt": f"{prefix_gcs_path}/ckpt/{model_name_unique}",
            "engine": f"{prefix_gcs_path}/engines/{model_name_unique}",
            "configs": f"{prefix_gcs_path}/configs/{model_name_unique}",
        }
    )


def set_directories(paths: dict[str, Path]) -> Result[None, str]:
    """Create necessary directories."""
    try:
        for key in ["ckpt", "engine"]:
            paths[key].mkdir(parents=True, exist_ok=True)
        logger.info("Directories created successfully")
        return Ok(None)
    except Exception as e:
        logger.error(f"Failed to setup directories: {str(e)}")
        return Err(str(e))


def set_environment(paths: dict[str, Path], model_name: str) -> Result[None, str]:
    """Set up environment variables and create necessary directories."""
    try:
        os.environ["MODEL_NAME"] = model_name
        os.environ["ENGINE_PATH"] = str(paths["engine"])
        os.environ["UNIFIED_CKPT_PATH"] = str(paths["ckpt"])
        os.environ["HF_LLAMA_MODEL"] = str(paths["hf_model"])

        logger.info("Environment setup completed successfully")
        return Ok(None)
    except Exception as e:
        logger.error(f"Failed to setup environment: {str(e)}")
        return Err(str(e))


def get_model_from_hf(model_name: str) -> Result[None, str]:
    """Download the model from Hugging Face asynchronously."""
    try:
        logger.info(f"Starting model download for {model_name}...")
        download_cmd = f"huggingface-cli download meta-llama/{model_name}"
        run_command(command=download_cmd).unwrap()
        logger.info("Model download completed")
        return Ok(None)
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return Err(str(e))


def set_checkpoint(paths: dict[str, Path]) -> Result[None, str]:
    """Convert the model checkpoint."""
    os.chdir(paths["backend"])
    try:
        convert_cmd = (
            f"/usr/bin/python3 tensorrt_llm/examples/models/core/llama/convert_checkpoint.py "
            f"--model_dir {paths['hf_model']} "
            f"--output_dir {paths['ckpt']} "
            f"--dtype float16"
        )
        run_command(command=convert_cmd).unwrap()
        logger.info("Checkpoint conversion completed")
        return Ok(None)
    except Exception as e:
        logger.error(f"Failed to convert checkpoint: {str(e)}")
        return Err(str(e))


def set_engine(paths: dict[str, Path]) -> Result[None, str]:
    """Build TensorRT-LLM model."""
    os.chdir(paths["backend"])
    try:
        build_cmd = (
            f"trtllm-build "
            f"--checkpoint_dir {paths['ckpt']} "
            f"--remove_input_padding enable "
            f"--gpt_attention_plugin float16 "
            f"--context_fmha enable "
            f"--gemm_plugin float16 "
            f"--output_dir {paths['engine']} "
            f"--kv_cache_type paged "
            f"--max_batch_size 64"
        )
        run_command(command=build_cmd).unwrap()
        logger.info("TensorRT-LLM build completed")
        return Ok(None)
    except Exception as e:
        logger.error(f"Failed to build TensorRT-LLM: {str(e)}")
        return Err(str(e))


def set_configs(paths: dict[str, Path]) -> Result[None, str]:
    """Prepare model configurations."""
    os.chdir(paths["backend"])
    try:
        # Copy config template
        run_command(command="cp all_models/inflight_batcher_llm/ llama_ifb -r").unwrap()

        # Fill templates
        prefix_cmd = "/usr/bin/python3 tools/fill_template.py"
        templates = [
            f"{prefix_cmd} -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:{paths['hf_model']},triton_max_batch_size:64,preprocessing_instance_count:1",
            f"{prefix_cmd} -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:{paths['hf_model']},triton_max_batch_size:64,postprocessing_instance_count:1",
            f"{prefix_cmd} -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32",
            f"{prefix_cmd} -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32",
            f"{prefix_cmd} -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:{paths['engine']},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32",
        ]

        for template in templates:
            run_command(command=template).unwrap()

        logger.info("Config preparation completed")
        return Ok(None)
    except Exception as e:
        logger.error(f"Failed to prepare configs: {str(e)}")
        return Err(str(e))


def save_output(
    paths: dict[str, Path], gcs_paths: dict[str, str], bucket_name: str
) -> Result[None, str]:
    """Save the model to google cloud storage."""
    save.save_dir(
        dir_path=paths["engine"], gcs_path=gcs_paths["engine"], bucket_name=bucket_name
    ).unwrap()
    save.save_dir(
        dir_path=paths["ckpt"], gcs_path=gcs_paths["ckpt"], bucket_name=bucket_name
    ).unwrap()
    save.save_dir(
        bucket_name=bucket_name,
        gcs_path=gcs_paths["configs"],
        dir_path=paths["backend"] / "llama_ifb",
    ).unwrap()
    return Ok(None)
