# Must-Read AI/Deep Learning/LLM Research Papers (2020-2025)

A curated list of essential research papers that have shaped modern AI, deep learning, and large language models over the past 5 years.

---

## Table of Contents
- [Foundation & Architecture](#foundation--architecture)
- [Large Language Models](#large-language-models)
- [Efficient Training & Fine-tuning](#efficient-training--fine-tuning)
- [Reasoning & Prompting](#reasoning--prompting)
- [Multimodal Models](#multimodal-models)
- [Diffusion Models](#diffusion-models)
- [Retrieval & RAG](#retrieval--rag)
- [Evaluation & Safety](#evaluation--safety)
- [Emerging Architectures](#emerging-architectures)

---

## Foundation & Architecture

### Attention is All You Need
**Authors:** Vaswani et al. (Google)
**Year:** 2017
**Link:** https://arxiv.org/abs/1706.03762

**Key Contributions:**
- Introduced the Transformer architecture, eliminating recurrence and relying entirely on attention mechanisms
- Proposed multi-head self-attention for parallel processing
- Revolutionized NLP and became the foundation for modern LLMs
- Demonstrated superior performance on translation tasks while being more parallelizable

**Why Read:** The foundational paper for all modern LLMs. Understanding transformers is essential for anyone working in AI.

---

### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)
**Authors:** Dosovitskiy et al. (Google Research)
**Year:** 2020
**Link:** https://arxiv.org/abs/2010.11929

**Key Contributions:**
- Applied pure transformer architecture to image classification
- Demonstrated that transformers can match or exceed CNN performance on vision tasks
- Introduced patch-based tokenization for images
- Showed the importance of pre-training on large datasets

**Why Read:** Proves transformers are not just for NLP, opening the door to unified architectures across modalities.

---

### Formal Algorithms for Transformers
**Authors:** Phuong & Hutter (DeepMind)
**Year:** 2022
**Link:** https://arxiv.org/abs/2207.09238

**Key Contributions:**
- Provides rigorous mathematical formalization of transformer architectures
- Covers attention mechanisms, positional encodings, and training procedures
- Bridges the gap between intuition and formal understanding
- Offers guarantees and proofs that underpin practical implementations

**Why Read:** For those who want to deeply understand the mathematics behind transformers.

---

## Large Language Models

### Language Models are Few-Shot Learners (GPT-3)
**Authors:** Brown et al. (OpenAI)
**Year:** 2020
**Link:** https://arxiv.org/abs/2005.14165

**Key Contributions:**
- Introduced GPT-3 with 175B parameters
- Demonstrated strong few-shot learning without fine-tuning
- Showed emergent abilities arise with scale
- Explored in-context learning capabilities across diverse tasks

**Why Read:** Marked the beginning of the modern LLM era and demonstrated the power of scale.

---

### Scaling Laws for Neural Language Models
**Authors:** Kaplan et al. (OpenAI)
**Year:** 2020
**Link:** https://arxiv.org/abs/2001.08361

**Key Contributions:**
- Established power-law relationships between model size, dataset size, and performance
- Provided predictive framework for model scaling
- Showed that larger models are more sample-efficient
- Guided resource allocation and training strategies for large-scale models

**Why Read:** Fundamental for understanding how to allocate resources in LLM training.

---

### GPT-4 Technical Report
**Authors:** OpenAI
**Year:** 2023
**Link:** https://arxiv.org/abs/2303.08774

**Key Contributions:**
- Documents multimodal capabilities and safety evaluations of a frontier LLM
- Highlights emergent reasoning, instruction-following, and tool-use behavior
- Provides insights into alignment techniques and evaluation methodologies
- Shares benchmarks spanning coding, reasoning, and professional exams

**Why Read:** Offers the most comprehensive public look at state-of-the-art LLM capabilities and limitations.

---

## Efficient Training & Fine-tuning

### Chinchilla: Training Compute-Optimal Large Language Models
**Authors:** Hoffmann et al. (DeepMind)
**Year:** 2022
**Link:** https://arxiv.org/abs/2203.15556

**Key Contributions:**
- Established that data quantity, not just parameter count, determines performance at scale
- Derived compute-optimal scaling laws balancing model size and training tokens
- Demonstrated that smaller models trained longer can outperform larger under-trained models
- Influenced training regimes for subsequent LLMs

**Why Read:** Essential for designing efficient training strategies that maximize compute investments.

---

### LoRA: Low-Rank Adaptation of Large Language Models
**Authors:** Hu et al. (Microsoft Research)
**Year:** 2021
**Link:** https://arxiv.org/abs/2106.09685

**Key Contributions:**
- Introduced parameter-efficient fine-tuning using low-rank decomposition
- Enabled adapting giant models on consumer hardware
- Maintained performance while updating a tiny fraction of weights
- Compatible with a wide range of transformer architectures

**Why Read:** A cornerstone technique for customizing LLMs without full retraining.

---

### QLoRA: Efficient Finetuning of Quantized LLMs
**Authors:** Dettmers et al. (University of Washington)
**Year:** 2023
**Link:** https://arxiv.org/abs/2305.14314

**Key Contributions:**
- Showed how to fine-tune 4-bit quantized LLMs without loss of quality
- Combined double quantization and paged optimizers for memory efficiency
- Demonstrated strong performance across benchmarks at a fraction of GPU costs
- Enabled widespread community fine-tuning on accessible hardware

**Why Read:** Critical for practitioners looking to deploy customized LLMs with constrained resources.

---

## Reasoning & Prompting

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
**Authors:** Wei et al. (Google Brain)
**Year:** 2022
**Link:** https://arxiv.org/abs/2201.11903

**Key Contributions:**
- Introduced chain-of-thought prompting to improve multi-step reasoning
- Demonstrated large accuracy gains on arithmetic and commonsense benchmarks
- Revealed that model scale correlates with reasoning benefits
- Provided a simple prompting technique adopted widely in practice

**Why Read:** Establishes the prompting paradigm that unlocked complex reasoning abilities in LLMs.

---

### Self-Consistency Improves Chain of Thought Reasoning in Language Models
**Authors:** Wang et al. (Google Research)
**Year:** 2022
**Link:** https://arxiv.org/abs/2203.11171

**Key Contributions:**
- Proposed sampling multiple reasoning paths and aggregating by majority vote
- Improved reliability and accuracy of chain-of-thought prompting
- Demonstrated robustness across mathematical and symbolic reasoning tasks
- Provided practical guidance for eliciting better answers from LLMs

**Why Read:** Offers a low-cost technique to stabilize and boost reasoning performance without retraining models.

---

### Toolformer: Language Models Can Teach Themselves to Use Tools
**Authors:** Schick et al. (Meta AI)
**Year:** 2023
**Link:** https://arxiv.org/abs/2302.04761

**Key Contributions:**
- Introduced self-supervised training for API calling and tool use
- Demonstrated improved reasoning by integrating external knowledge sources
- Showed automated dataset creation for tool annotations
- Highlighted pathways for augmenting LLMs with plug-in capabilities

**Why Read:** Illustrates how LLMs can extend their abilities through autonomous tool discovery and usage.

---

## Multimodal Models

### CLIP: Learning Transferable Visual Models From Natural Language Supervision
**Authors:** Radford et al. (OpenAI)
**Year:** 2021
**Link:** https://arxiv.org/abs/2103.00020

**Key Contributions:**
- Paired large-scale image-text contrastive learning
- Enabled zero-shot classification across diverse visual tasks
- Showed that internet-scale data can replace labeled datasets
- Provided embeddings that power many downstream multimodal systems

**Why Read:** A landmark in bridging vision and language using a shared representation space.

---

### Flamingo: A Visual Language Model for Few-Shot Learning
**Authors:** Alayrac et al. (DeepMind)
**Year:** 2022
**Link:** https://arxiv.org/abs/2204.14198

**Key Contributions:**
- Combined frozen language models with adaptive cross-attention visual encoders
- Delivered state-of-the-art few-shot performance on multimodal benchmarks
- Supported open-ended dialogue grounded in images and video
- Introduced lightweight adapters that preserve pre-trained knowledge

**Why Read:** Demonstrates how to extend language models into powerful multimodal learners.

---

### Gemini: A Family of Highly Capable Multimodal Models
**Authors:** Team Google DeepMind
**Year:** 2023
**Link:** https://arxiv.org/abs/2312.11805

**Key Contributions:**
- Unified training across text, image, audio, and video modalities
- Showcased strong reasoning, coding, and perception capabilities
- Highlighted safety alignment and evaluation for multimodal systems
- Provided scaling insights for next-generation multimodal AI

**Why Read:** Offers a blueprint for the latest multimodal frontier models and their deployment considerations.

---

## Diffusion Models

### Denoising Diffusion Probabilistic Models (DDPM)
**Authors:** Ho et al. (Google Brain)
**Year:** 2020
**Link:** https://arxiv.org/abs/2006.11239

**Key Contributions:**
- Introduced diffusion probabilistic models for high-quality image synthesis
- Demonstrated competitive performance with autoregressive and GAN models
- Provided a simple training objective with stable optimization
- Sparked the diffusion model renaissance across modalities

**Why Read:** Lays the groundwork for modern diffusion-based generative modeling.

---

### High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)
**Authors:** Rombach et al. (LMU Munich & Stability AI)
**Year:** 2022
**Link:** https://arxiv.org/abs/2112.10752

**Key Contributions:**
- Introduced latent diffusion for efficient high-resolution generation
- Enabled text-to-image synthesis with reduced compute requirements
- Released open models that catalyzed widespread experimentation
- Demonstrated controllable generation via conditioning mechanisms

**Why Read:** The foundation for practical, open diffusion systems used across industry and research.

---

### Imagen: Text-to-Image Diffusion Models with Large Language Models
**Authors:** Saharia et al. (Google Research)
**Year:** 2022
**Link:** https://arxiv.org/abs/2205.11487

**Key Contributions:**
- Combined large language model text embeddings with diffusion decoders
- Achieved state-of-the-art fidelity and alignment on text-to-image benchmarks
- Demonstrated superior sample quality through cascaded diffusion stages
- Highlighted the importance of language understanding for image synthesis

**Why Read:** Illustrates how LLM advances enhance diffusion-based generative models.

---

## Retrieval & RAG

### Retrieval-Augmented Generation for Knowledge-Intensive NLP
**Authors:** Lewis et al. (Facebook AI Research)
**Year:** 2020
**Link:** https://arxiv.org/abs/2005.11401

**Key Contributions:**
- Introduced the RAG framework combining retrieval and generation end-to-end
- Improved factual accuracy on open-domain QA tasks
- Demonstrated hybrid models outperforming pure generative baselines
- Established a template for production retrieval-augmented systems

**Why Read:** The seminal work that defines RAG architectures used widely in industry applications.

---

### Atlas: Few-shot Learning with Retrieval Augmented Language Models
**Authors:** Izacard et al. (Meta AI & UCL)
**Year:** 2022
**Link:** https://arxiv.org/abs/2208.03299

**Key Contributions:**
- Leveraged large-scale retrieval databases to enhance few-shot generalization
- Demonstrated state-of-the-art results on QA and reasoning benchmarks
- Explored scaling behaviors for retrieval-augmented models
- Provided ablations on retriever and generator interplay

**Why Read:** Shows how retrieval boosts sample efficiency and accuracy for LLMs in data-scarce settings.

---

### RETRO: Improving Language Models with Real Retrieval
**Authors:** Borgeaud et al. (DeepMind)
**Year:** 2022
**Link:** https://arxiv.org/abs/2112.04426

**Key Contributions:**
- Augmented transformers with nearest-neighbor retrieval during training and inference
- Demonstrated reduced parameter counts while maintaining performance
- Introduced efficient indexing and chunking strategies for large corpora
- Validated that retrieval can replace scale for certain capabilities

**Why Read:** Provides practical insights into marrying retrieval databases with generative modeling at scale.

---

## Evaluation & Safety

### TruthfulQA: Measuring How Models Mimic Human Falsehoods
**Authors:** Lin et al. (OpenAI & UC Berkeley)
**Year:** 2021
**Link:** https://arxiv.org/abs/2109.07958

**Key Contributions:**
- Created a benchmark for factuality and resistance to misinformation
- Evaluated models on their tendency to imitate human falsehoods
- Highlighted limitations of prompt-based mitigation strategies
- Provided diagnostic tasks for alignment research

**Why Read:** Establishes a rigorous benchmark for evaluating truthfulness in language models.

---

### Constitutional AI: Harmlessness from AI Feedback
**Authors:** Bai et al. (Anthropic)
**Year:** 2022
**Link:** https://arxiv.org/abs/2212.08073

**Key Contributions:**
- Introduced AI feedback loops guided by a set of constitutional principles
- Reduced reliance on human annotators for alignment training
- Demonstrated improvements in helpfulness while reducing harmful outputs
- Provided a scalable recipe for safety-aligned instruction tuning

**Why Read:** A landmark approach for aligning LLM behavior using AI-generated feedback.

---

### Holistic Evaluation of Language Models (HELM)
**Authors:** Liang et al. (Stanford Center for Research on Foundation Models)
**Year:** 2022
**Link:** https://arxiv.org/abs/2211.09110

**Key Contributions:**
- Proposed a comprehensive evaluation framework covering accuracy, robustness, and fairness
- Benchmarked numerous LLMs across tasks, scenarios, and metrics
- Highlighted trade-offs and failure modes in model deployment
- Provided standardized protocols for transparent reporting

**Why Read:** Offers a rigorous methodology for assessing foundation models across dimensions beyond accuracy.

---

## Emerging Architectures

### Mixture-of-Depths: Dynamically Gated Deep Networks
**Authors:** Bapna et al. (Google Research)
**Year:** 2024
**Link:** https://arxiv.org/abs/2401.02038

**Key Contributions:**
- Introduced dynamic depth selection to allocate compute adaptively
- Achieved efficiency gains without sacrificing performance
- Demonstrated benefits across language and multimodal tasks
- Presented a general recipe for conditional computation in transformers

**Why Read:** Points toward future architectures that scale depth flexibly based on input complexity.

---

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces
**Authors:** Gu & Dao (Princeton & CMU)
**Year:** 2023
**Link:** https://arxiv.org/abs/2312.00752

**Key Contributions:**
- Proposed selective state space models achieving linear-time sequence processing
- Matched or exceeded transformer performance on language and vision benchmarks
- Showed efficiency gains for long-context tasks
- Offered an alternative architecture for scaling beyond attention bottlenecks

**Why Read:** Highlights promising directions for non-transformer sequence models with competitive capability.

---

### HyperAttention: Long-context Attention in Near-Linear Time
**Authors:** Dao et al. (Stanford & Together AI)
**Year:** 2024
**Link:** https://arxiv.org/abs/2405.12981

**Key Contributions:**
- Developed a provably efficient attention mechanism for contexts exceeding 1M tokens
- Combined sparse factorization with hardware-aware kernels
- Demonstrated state-of-the-art long-context modeling on academic benchmarks
- Provided open-source implementations for practical adoption

**Why Read:** Essential for practitioners tackling long-context reasoning and retrieval-heavy workloads.

---
