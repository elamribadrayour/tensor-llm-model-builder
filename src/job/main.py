import os
import shutil
import asyncio
import subprocess
from uuid import uuid4
from pathlib import Path

from loguru import logger
from result import Err, Ok, Result
import google.cloud.storage as storage
from huggingface_hub import hf_hub_download


class ModelBuilder:
    def __init__(self: "ModelBuilder") -> None:
        """Initialize the ModelBuilder class."""
        if "MODEL_NAME" not in os.environ:
            raise Exception("MODEL_NAME environment variable is not set")
        if "HF_TOKEN" not in os.environ:
            raise Exception("HF_TOKEN environment variable is not set")
        if "HUGGINGFACE_HUB_CACHE" not in os.environ:
            raise Exception("HUGGINGFACE_HUB_CACHE environment variable is not set")

        self.base_path = Path("/data")
        if not self.base_path.exists():
            raise Exception(
                f"Base path {self.base_path} does not exist, a pvc of at least 1Ti mounted on /data is required because of large file sizes"
            )

        self.model_name = os.environ["MODEL_NAME"]
        self.model_name_unique = f"{self.model_name}-{uuid4()}"
        config_path = hf_hub_download(
            "meta-llama/" + self.model_name, filename="config.json"
        )
        self.hf_model_path = str(Path(config_path).parent)

        self.ckpt_path = self.base_path / "tensorrt" / "ckpt" / self.model_name_unique
        self.engine_path = (
            self.base_path / "tensorrt" / "engines" / self.model_name_unique
        )
        self.backend_path = self.base_path / "tensorrtllm_backend"
        if self.backend_path.exists():
            shutil.rmtree(self.backend_path)

    @staticmethod
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

    @staticmethod
    async def run_command_async(
        command: str, cwd: str | None = None
    ) -> Result[bool, str]:
        """Execute a shell command asynchronously and handle errors."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: ModelBuilder.run_command(command, cwd)
            )
        except Exception as e:
            return Err(str(e))

    def set_environment(self: "ModelBuilder") -> Result[None, str]:
        """Set up environment variables and create necessary directories."""
        try:
            os.environ["MODEL_NAME"] = self.model_name
            os.environ["ENGINE_PATH"] = str(self.engine_path)
            os.environ["UNIFIED_CKPT_PATH"] = str(self.ckpt_path)

            for path in [self.engine_path, self.ckpt_path]:
                path.mkdir(parents=True, exist_ok=True)

            os.environ["HF_LLAMA_MODEL"] = self.hf_model_path

            logger.info("Environment setup completed successfully")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to setup environment: {str(e)}")
            return Err(str(e))

    async def set_tensorrtllm_backend(self: "ModelBuilder") -> Result[None, str]:
        """Set up TensorRT-LLM backend asynchronously."""
        try:
            os.chdir(self.base_path)
            result = await ModelBuilder.run_command_async(command="git lfs install")
            result.unwrap()

            result = await ModelBuilder.run_command_async(
                command="git clone https://github.com/triton-inference-server/tensorrtllm_backend.git"
            )
            result.unwrap()

            os.chdir(self.backend_path)
            result = await ModelBuilder.run_command_async(
                command="git submodule update --init --recursive"
            )
            result.unwrap()

            logger.info("TensorRT-LLM backend setup completed")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to setup TensorRT-LLM backend: {str(e)}")
            return Err(str(e))

    async def get_model_from_hf(self: "ModelBuilder") -> Result[None, str]:
        """Download the model from Hugging Face asynchronously."""
        try:
            logger.info(f"Starting model download for {self.model_name}...")
            download_cmd = f"huggingface-cli download meta-llama/{self.model_name}"
            result = await ModelBuilder.run_command_async(command=download_cmd)
            result.unwrap()
            logger.info("Model download completed")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            return Err(str(e))

    def set_checkpoint(self: "ModelBuilder") -> Result[None, str]:
        """Convert the model checkpoint."""
        os.chdir(self.backend_path)
        try:
            convert_cmd = (
                f"/usr/bin/python3 tensorrt_llm/examples/models/core/llama/convert_checkpoint.py "
                f"--model_dir {self.hf_model_path} "
                f"--output_dir {self.ckpt_path} "
                f"--dtype float16"
            )
            ModelBuilder.run_command(command=convert_cmd).unwrap()
            logger.info("Checkpoint conversion completed")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to convert checkpoint: {str(e)}")
            return Err(str(e))

    def set_engine(self: "ModelBuilder") -> Result[None, str]:
        """Build TensorRT-LLM model."""
        os.chdir(self.backend_path)
        try:
            build_cmd = (
                f"trtllm-build "
                f"--checkpoint_dir {self.ckpt_path} "
                f"--remove_input_padding enable "
                f"--gpt_attention_plugin float16 "
                f"--context_fmha enable "
                f"--gemm_plugin float16 "
                f"--output_dir {self.engine_path} "
                f"--kv_cache_type paged "
                f"--max_batch_size 64"
            )
            ModelBuilder.run_command(command=build_cmd).unwrap()
            logger.info("TensorRT-LLM build completed")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to build TensorRT-LLM: {str(e)}")
            return Err(str(e))

    def set_configs(self: "ModelBuilder") -> Result[None, str]:
        """Prepare model configurations."""
        os.chdir(self.backend_path)
        try:
            # Copy config template
            ModelBuilder.run_command(
                command="cp all_models/inflight_batcher_llm/ llama_ifb -r"
            ).unwrap()

            # Fill templates
            prefix_cmd = "/usr/bin/python3 tools/fill_template.py"
            templates = [
                f"{prefix_cmd} -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:{self.hf_model_path},triton_max_batch_size:64,preprocessing_instance_count:1",
                f"{prefix_cmd} -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:{self.hf_model_path},triton_max_batch_size:64,postprocessing_instance_count:1",
                f"{prefix_cmd} -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32",
                f"{prefix_cmd} -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32",
                f"{prefix_cmd} -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:{self.engine_path},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32",
            ]

            for template in templates:
                ModelBuilder.run_command(command=template).unwrap()

            logger.info("Config preparation completed")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to prepare configs: {str(e)}")
            return Err(str(e))

    def save(self: "ModelBuilder") -> Result[None, str]:
        """Save the model to google cloud storage."""
        client = storage.Client()
        bucket = client.bucket(bucket_name="datascience-result-prod")
        prefix_gcs_path = "tensorrt/models"

        logger.info("Saving engine to GCS")
        engine_gcs_path = f"{prefix_gcs_path}/engines/{self.model_name_unique}/"
        for file_path in Path(self.engine_path).glob("*"):
            logger.info(f"Saving {file_path.name} to GCS")
            blob = bucket.blob(blob_name=f"{engine_gcs_path}/{file_path.name}")
            blob.upload_from_filename(filename=file_path)

        logger.info("Saving checkpoint to GCS")
        checkpoint_gcs_path = f"{prefix_gcs_path}/checkpoints/{self.model_name_unique}/"
        for file_path in Path(self.ckpt_path).glob("*"):
            logger.info(f"Saving {file_path.name} to GCS")
            blob = bucket.blob(blob_name=f"{checkpoint_gcs_path}/{file_path.name}")
            blob.upload_from_filename(filename=file_path)

        logger.info("Saving configs to GCS")
        configs_gcs_path = f"{prefix_gcs_path}/configs/{self.model_name_unique}/"
        for file_path in Path(self.base_path / "llama_ifb").glob("*"):
            logger.info(f"Saving {file_path.name} to GCS")
            blob = bucket.blob(blob_name=f"{configs_gcs_path}/{file_path.name}")
            blob.upload_from_filename(filename=file_path)

        return Ok(None)

    async def get_requirements(self: "ModelBuilder") -> Result[None, str]:
        """Run backend setup and model download in parallel."""
        logger.info("Starting parallel tasks: backend setup and model download")
        backend_task = asyncio.create_task(self.set_tensorrtllm_backend())
        model_task = asyncio.create_task(self.get_model_from_hf())

        logger.info("Both tasks created, waiting for completion...")
        tasks = await asyncio.gather(backend_task, model_task)
        for task in tasks:
            task.unwrap()
        return Ok(None)

    def run(self: "ModelBuilder") -> Result[None, str]:
        """Main build process."""
        logger.info("getting python requirements")
        try:
            logger.info(f"Starting build process for {self.model_name}")
            self.set_environment().unwrap()

            asyncio.run(self.get_requirements()).unwrap()

            self.set_checkpoint().unwrap()
            self.set_engine().unwrap()
            self.set_configs().unwrap()
            self.save().unwrap()
            logger.info(f"Build process completed successfully for {self.model_name}")
            return Ok(None)
        except Exception as e:
            logger.error(f"Build process failed: {str(e)}")
            return Err(str(e))


if __name__ == "__main__":
    ModelBuilder().run().unwrap()
