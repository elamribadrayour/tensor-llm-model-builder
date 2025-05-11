import subprocess

import asyncio
from loguru import logger
from result import Err, Ok, Result


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


async def run_command_async(command: str, cwd: str | None = None) -> Result[bool, str]:
    """Execute a shell command asynchronously and handle errors."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: run_command(command, cwd))
    except Exception as e:
        return Err(str(e))
