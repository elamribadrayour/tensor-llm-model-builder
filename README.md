# TensorRT LLM Model Builder

A Python-based tool for building and managing TensorRT-optimized Large Language Models (LLMs).

## Overview

This project provides a streamlined way to build and optimize LLM models using NVIDIA's TensorRT framework. It includes utilities for model conversion, optimization, and deployment, with support for Hugging Face models and Google Cloud Storage integration.

## Features

- TensorRT optimization for LLM models
- Hugging Face model integration
- Google Cloud Storage support
- Docker-based deployment
- Comprehensive logging with Loguru
- Type-safe error handling with Result

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support
- Docker and Docker Compose (for containerized deployment)

## Installation

Clone the repository:
```bash
git clone https://github.com/elamribadrayour/tensorrt-llm-model-builder.git
cd tensorrt-llm-model-builder
```

Create and configure your environment variables:
```bash
cp .env.example .env
```

Fill in the required environment variables in `.env`:
```env
HUGGINGFACE_TOKEN=your-huggingface-token
```

## Usage

### Local Development

1. Set up your Python environment:
```bash
uv sync --frozen
```

### Docker Deployment

1. Build and run using Docker Compose:
```bash
docker-compose up --build
```

## Project Structure

```
.
├── src/            # Source code
├── models/         # Model storage
├── Dockerfile     # Container definition
└── docker-compose.yml
```

## License

See [LICENSE](LICENSE) file for details.
