use std::collections::HashMap;

pub struct ModelConfig {
    pub name: &'static str,
    pub onnx_url: &'static str,
    pub tokenizer_url: &'static str,
    pub filename: &'static str,
    pub dimension: usize,
    pub is_reranker: bool,
}

pub fn get_model_registry() -> HashMap<&'static str, ModelConfig> {
    let mut m = HashMap::new();

    m.insert(
        "Xenova/bge-m3",
        ModelConfig {
            name: "Xenova/bge-m3",
            onnx_url: "https://huggingface.co/Xenova/bge-m3/resolve/main/onnx/model_quantized.onnx",
            tokenizer_url: "https://huggingface.co/Xenova/bge-m3/resolve/main/tokenizer.json",
            filename: "bge-m3-q4.onnx",
            dimension: 1024,
            is_reranker: false,
        },
    );

    m.insert("BAAI/bge-small-en-v1.5", ModelConfig {
        name: "BAAI/bge-small-en-v1.5",
        onnx_url: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json",
        filename: "bge-small-en-v1.5-q4.onnx",
        dimension: 384,
        is_reranker: false,
    });

    m.insert("Xenova/bge-small-en-v1.5", ModelConfig {
        name: "Xenova/bge-small-en-v1.5",
        onnx_url: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json",
        filename: "bge-small-en-v1.5-q4.onnx",
        dimension: 384,
        is_reranker: false,
    });

    m.insert("BAAI/bge-base-en-v1.5", ModelConfig {
        name: "BAAI/bge-base-en-v1.5",
        onnx_url: "https://huggingface.co/Xenova/bge-base-en-v1.5/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/bge-base-en-v1.5/resolve/main/tokenizer.json",
        filename: "bge-base-en-v1.5-q4.onnx",
        dimension: 768,
        is_reranker: false,
    });

    m.insert("cross-encoder/ms-marco-MiniLM-L-6-v2", ModelConfig {
        name: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        onnx_url: "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json",
        filename: "ms-marco-MiniLM-L-6-v2-q4.onnx",
        dimension: 1,
        is_reranker: true,
    });

    m.insert("Xenova/bge-reranker-base", ModelConfig {
        name: "Xenova/bge-reranker-base",
        onnx_url: "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/tokenizer.json",
        filename: "bge-reranker-base-q4.onnx",
        dimension: 1,
        is_reranker: true,
    });

    m.insert("Xenova/bge-reranker-v2-m3", ModelConfig {
        name: "Xenova/bge-reranker-v2-m3",
        onnx_url: "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/tokenizer.json",
        filename: "bge_reranker_v2_m3_model_quantized.onnx",
        dimension: 1,
        is_reranker: true,
    });

    m.insert("Xenova/bge-reranker-v2-gemma", ModelConfig {
        name: "Xenova/bge-reranker-v2-gemma",
        onnx_url: "https://huggingface.co/onnx-community/bge-reranker-v2-gemma-ONNX/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/onnx-community/bge-reranker-v2-gemma-ONNX/resolve/main/tokenizer.json",
        filename: "bge_reranker_v2_gemma_model_quantized.onnx",
        dimension: 1,
        is_reranker: true,
    });

    m.insert("Xenova/jina-reranker-v2-base-multilingual", ModelConfig {
        name: "Xenova/jina-reranker-v2-base-multilingual",
        onnx_url: "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/onnx/model_int8.onnx",
        tokenizer_url: "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/tokenizer.json",
        filename: "jina_reranker_v2_model_int8.onnx",
        dimension: 1,
        is_reranker: true,
    });

    m.insert("Xenova/jina-embeddings-v2-base-code", ModelConfig {
        name: "Xenova/jina-embeddings-v2-base-code",
        onnx_url: "https://huggingface.co/Xenova/jina-embeddings-v2-base-code/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/jina-embeddings-v2-base-code/resolve/main/tokenizer.json",
        filename: "jina_code_v2_model_quantized.onnx",
        dimension: 768,
        is_reranker: false,
    });

    m.insert("IBM/granite-embedding-english-r2", ModelConfig {
        name: "IBM/granite-embedding-english-r2",
        onnx_url: "https://huggingface.co/onnx-community/granite-embedding-english-r2-ONNX/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/onnx-community/granite-embedding-english-r2-ONNX/resolve/main/tokenizer.json",
        filename: "IBM_granite_r2_model_quantized.onnx",
        dimension: 768,
        is_reranker: false,
    });

    m
}
