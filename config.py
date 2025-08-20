# Multi-Agent DSA System Configuration

# Ollama Configuration
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = "http://localhost:11434"

# Fine-tuned Models Configuration
MODELS = {
    "analyzer": {
        "name": "llama3.1-dsa-analyzer",
        "base_model": "llama3.1:8b",
        "fine_tuned": False,
        "description": "Specialized for DSA problem analysis"
    },
    "coder": {
        "name": "llama3.1-dsa-coder",
        "base_model": "llama3.1:8b", 
        "fine_tuned": True,
        "peft_adapter": "dsa-coding-adapter",
        "description": "Fine-tuned with PEFT and Unsloth for Python DSA coding"
    },
    "tester": {
        "name": "llama3.1-dsa-tester",
        "base_model": "llama3.1:8b",
        "fine_tuned": False,
        "description": "Specialized for code testing and validation"
    }
}

# PEFT (Parameter-Efficient Fine-Tuning) Configuration
PEFT_CONFIG = {
    "technique": "LoRA",  # Low-Rank Adaptation
    "rank": 16,
    "alpha": 32,
    "dropout": 0.1,
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
}

# Unsloth Configuration for Training
UNSLOTH_CONFIG = {
    "max_seq_length": 4096,
    "dtype": "float16",
    "load_in_4bit": True,
    "use_gradient_checkpointing": True,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 100,
    "save_steps": 500,
    "logging_steps": 10
}

# Agent Configuration
AGENT_CONFIG = {
    "max_rounds": 15,
    "timeout": 300,
    "temperature": 0.3,
    "max_tokens": 2048,
    "seed": 42
}

# DSA Categories and Weights
DSA_CATEGORIES = {
    "arrays": 0.20,
    "strings": 0.15,
    "linked_lists": 0.10,
    "trees": 0.15,
    "graphs": 0.15,
    "dynamic_programming": 0.15,
    "sorting": 0.05,
    "searching": 0.05
}

# Success Rate Targets
SUCCESS_TARGETS = {
    "easy": 0.85,
    "medium": 0.70,
    "hard": 0.55,
    "overall": 0.70
}

# Test Configuration
TEST_CONFIG = {
    "timeout_seconds": 30,
    "max_memory_mb": 512,
    "enable_profiling": True,
    "generate_additional_tests": True,
    "test_edge_cases": True
}