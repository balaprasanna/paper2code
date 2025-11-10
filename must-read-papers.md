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
- Explored in-context learning capabilities

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
- Influenced the "scaling is all you need" paradigm

**Why Read:** Fundamental for understanding how to allocate resources in LLM training.

---

### Training language models to follow instructions with human feedback (InstructGPT)
**Authors:** Ouyang et al. (OpenAI)
**Year:** 2022
**Link:** https://arxiv.org/abs/2203.02155

**Key Contributions:**
- Introduced RLHF (Reinforcement Learning from Human Feedback)
- Demonstrated alignment techniques for making LLMs more helpful and safe
- Showed smaller aligned models can outperform larger unaligned ones
- Became the foundation for ChatGPT

**Why Read:** Critical for understanding how modern conversational AI systems are trained.

---

### PaLM: Scaling Language Modeling with Pathways
**Authors:** Chowdhery et al. (Google)
**Year:** 2022
**Link:** https://arxiv.org/abs/2204.02311

**Key Contributions:**
- Introduced 540B parameter model trained on 6144 TPU chips
- Demonstrated breakthrough performance on reasoning tasks
- Showed continued scaling benefits
- Introduced efficient training infrastructure

**Why Read:** Represents state-of-the-art in scaling and efficient training.

---

### LLaMA: Open and Efficient Foundation Language Models
**Authors:** Touvron et al. (Meta)
**Year:** 2023
**Link:** https://arxiv.org/abs/2302.13971

**Key Contributions:**
- Released collection of models from 7B to 65B parameters
- Demonstrated that smaller models trained on more data can match larger models
- Emphasized training compute-optimal models
- Kickstarted the open-source LLM movement

**Why Read:** Changed the landscape by making powerful LLMs accessible to researchers.

---

### Llama 2: Open Foundation and Fine-Tuned Chat Models
**Authors:** Touvron et al. (Meta)
**Year:** 2023
**Link:** https://arxiv.org/abs/2307.09288

**Key Contributions:**
- Released improved and commercially usable LLaMA models
- Detailed RLHF training methodology
- Comprehensive safety evaluations and red-teaming
- Provided both base and chat-tuned models

**Why Read:** Comprehensive look at building production-ready open-source LLMs.

---

### GPT-4 Technical Report
**Authors:** OpenAI
**Year:** 2023
**Link:** https://arxiv.org/abs/2303.08774

**Key Contributions:**
- Multimodal model accepting both text and images
- Demonstrated human-level performance on various benchmarks
- Improved steerability and safety
- Discussed alignment techniques and limitations

**Why Read:** Represents the current frontier in LLM capabilities.

---

### Constitutional AI: Harmlessness from AI Feedback
**Authors:** Bai et al. (Anthropic)
**Year:** 2022
**Link:** https://arxiv.org/abs/2212.08073

**Key Contributions:**
- Introduced AI-based feedback for alignment instead of purely human feedback
- Proposed constitution-based training for value alignment
- Reduced reliance on human supervision
- Improved scalable oversight techniques

**Why Read:** Important alternative approach to RLHF for AI alignment.

---

## Efficient Training & Fine-tuning

### LoRA: Low-Rank Adaptation of Large Language Models
**Authors:** Hu et al. (Microsoft)
**Year:** 2021
**Link:** https://arxiv.org/abs/2106.09685

**Key Contributions:**
- Introduced parameter-efficient fine-tuning via low-rank matrices
- Reduces trainable parameters by 10,000x
- Maintains comparable performance to full fine-tuning
- Enables multiple task-specific adaptations

**Why Read:** Essential for practical LLM fine-tuning with limited resources.

---

### QLoRA: Efficient Finetuning of Quantized LLMs
**Authors:** Dettmers et al.
**Year:** 2023
**Link:** https://arxiv.org/abs/2305.14314

**Key Contributions:**
- Combines quantization with LoRA for extreme memory efficiency
- Enables fine-tuning 65B models on a single GPU
- Introduces 4-bit NormalFloat (NF4) quantization
- Democratizes LLM fine-tuning

**Why Read:** Makes fine-tuning large models accessible on consumer hardware.

---

### FlashAttention: Fast and Memory-Efficient Exact Attention
**Authors:** Dao et al. (Stanford)
**Year:** 2022
**Link:** https://arxiv.org/abs/2205.14135

**Key Contributions:**
- IO-aware attention algorithm reducing memory from quadratic to linear
- 2-4x speedup in transformer training
- Enables longer context lengths
- Exact attention (no approximation)

**Why Read:** Fundamental optimization enabling efficient long-context models.

---

### FlashAttention-2: Faster Attention with Better Parallelism
**Authors:** Dao (Together AI)
**Year:** 2023
**Link:** https://arxiv.org/abs/2307.08691

**Key Contributions:**
- 2x speedup over FlashAttention
- Better GPU utilization through improved parallelism
- Work partitioning improvements
- Further reduces memory overhead

**Why Read:** Current state-of-the-art in efficient attention computation.

---

## Reasoning & Prompting

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
**Authors:** Wei et al. (Google)
**Year:** 2022
**Link:** https://arxiv.org/abs/2201.11903

**Key Contributions:**
- Showed that prompting models to show intermediate steps dramatically improves reasoning
- Simple technique applicable to existing models
- Demonstrated emergent ability in sufficiently large models
- Became foundational prompting technique

**Why Read:** Revolutionary prompting method that unlocked reasoning capabilities in LLMs.

---

### Self-Consistency Improves Chain of Thought Reasoning
**Authors:** Wang et al. (Google)
**Year:** 2022
**Link:** https://arxiv.org/abs/2203.11171

**Key Contributions:**
- Sample multiple reasoning paths and take majority vote
- Significant improvements over standard CoT
- No additional training required
- Works across various reasoning tasks

**Why Read:** Simple but powerful improvement to chain-of-thought reasoning.

---

### ReAct: Synergizing Reasoning and Acting in Language Models
**Authors:** Yao et al. (Princeton, Google)
**Year:** 2022
**Link:** https://arxiv.org/abs/2210.03629

**Key Contributions:**
- Interleaves reasoning traces with task-specific actions
- Enables LLMs to interact with external environments
- Foundation for agent-based systems
- Improved interpretability through reasoning traces

**Why Read:** Key paper for building LLM-powered agents and tools.

---

### Tree of Thoughts: Deliberate Problem Solving with Large Language Models
**Authors:** Yao et al. (Princeton, Google DeepMind)
**Year:** 2023
**Link:** https://arxiv.org/abs/2305.10601

**Key Contributions:**
- Generalizes chain-of-thought to tree-structured exploration
- Enables look-ahead and backtracking
- Self-evaluation of reasoning steps
- Solves problems requiring strategic planning

**Why Read:** Advanced reasoning framework for complex problem-solving.

---

## Multimodal Models

### Learning Transferable Visual Models From Natural Language Supervision (CLIP)
**Authors:** Radford et al. (OpenAI)
**Year:** 2021
**Link:** https://arxiv.org/abs/2103.00020

**Key Contributions:**
- Trained vision models using natural language supervision
- Zero-shot transfer to various image tasks
- Contrastive learning on 400M image-text pairs
- Foundation for multimodal understanding

**Why Read:** Revolutionized vision-language models and zero-shot learning.

---

### Flamingo: a Visual Language Model for Few-Shot Learning
**Authors:** Alayrac et al. (DeepMind)
**Year:** 2022
**Link:** https://arxiv.org/abs/2204.14198

**Key Contributions:**
- Interleaved vision and language processing
- Few-shot learning on multimodal tasks
- Cross-attention between vision and language
- Strong performance with minimal task-specific data

**Why Read:** Important architecture for multimodal few-shot learning.

---

### Visual Instruction Tuning (LLaVA)
**Authors:** Liu et al. (Microsoft, University of Wisconsin-Madison)
**Year:** 2023
**Link:** https://arxiv.org/abs/2304.08485

**Key Contributions:**
- Open-source multimodal conversational agent
- Instruction tuning for vision-language models
- GPT-assisted visual instruction data generation
- Impressive performance with efficient training

**Why Read:** Democratized multimodal LLMs with open-source approach.

---

### GPT-4V(ision) System Card
**Authors:** OpenAI
**Year:** 2023
**Link:** https://openai.com/research/gpt-4v-system-card

**Key Contributions:**
- Detailed analysis of GPT-4's vision capabilities
- Safety evaluations for multimodal systems
- Real-world use cases and limitations
- Risk assessments and mitigations

**Why Read:** Understanding capabilities and safety considerations of frontier multimodal models.

---

## Diffusion Models

### Denoising Diffusion Probabilistic Models (DDPM)
**Authors:** Ho et al. (Google Brain)
**Year:** 2020
**Link:** https://arxiv.org/abs/2006.11239

**Key Contributions:**
- Formalized diffusion models for high-quality image generation
- Iterative denoising process
- Matched or exceeded GAN quality
- Foundation for modern generative models

**Why Read:** The foundational paper for understanding diffusion-based generation.

---

### High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)
**Authors:** Rombach et al. (LMU Munich)
**Year:** 2022
**Link:** https://arxiv.org/abs/2112.10752

**Key Contributions:**
- Diffusion in latent space rather than pixel space
- Dramatically reduced computational requirements
- Cross-attention for conditioning (text, images)
- Enabled open-source high-quality image generation

**Why Read:** Made diffusion models practical and accessible, powering Stable Diffusion.

---

### Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2)
**Authors:** Ramesh et al. (OpenAI)
**Year:** 2022
**Link:** https://arxiv.org/abs/2204.06125

**Key Contributions:**
- CLIP-guided diffusion for text-to-image generation
- Prior network mapping text to image embeddings
- Hierarchical generation for high resolution
- Strong compositional understanding

**Why Read:** Demonstrated the power of combining CLIP with diffusion for creative generation.

---

## Retrieval & RAG

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)
**Authors:** Lewis et al. (Facebook AI)
**Year:** 2020
**Link:** https://arxiv.org/abs/2005.11401

**Key Contributions:**
- Combined retrieval with generation for factual accuracy
- Fine-tunable retriever and generator
- Improved performance on knowledge-intensive tasks
- Reduced hallucinations

**Why Read:** Foundational approach for grounding LLMs in external knowledge.

---

### In-Context Retrieval-Augmented Language Models
**Authors:** Ram et al. (AI21 Labs)
**Year:** 2023
**Link:** https://arxiv.org/abs/2302.00083

**Key Contributions:**
- Training-free retrieval augmentation
- Documents inserted into context
- Improved factuality and up-to-date information
- Practical deployment considerations

**Why Read:** Practical approach to enhancing LLMs with retrieval without retraining.

---

## Evaluation & Safety

### Measuring Massive Multitask Language Understanding (MMLU)
**Authors:** Hendrycks et al. (UC Berkeley)
**Year:** 2021
**Link:** https://arxiv.org/abs/2009.03300

**Key Contributions:**
- Comprehensive benchmark across 57 subjects
- Tests knowledge from elementary to professional level
- Became standard evaluation for LLMs
- Measures world knowledge and reasoning

**Why Read:** Understanding how LLMs are evaluated on knowledge and reasoning.

---

### TruthfulQA: Measuring How Models Mimic Human Falsehoods
**Authors:** Lin et al. (OpenAI, University of Oxford)
**Year:** 2022
**Link:** https://arxiv.org/abs/2109.07958

**Key Contributions:**
- Benchmark for evaluating truthfulness
- Questions designed to test common misconceptions
- Showed larger models can be less truthful
- Important for alignment research

**Why Read:** Critical perspective on LLM reliability and truthfulness.

---

### Red Teaming Language Models to Reduce Harms
**Authors:** Ganguli et al. (Anthropic)
**Year:** 2022
**Link:** https://arxiv.org/abs/2209.07858

**Key Contributions:**
- Systematic approach to finding model vulnerabilities
- Red team attack strategies and defenses
- Analysis of harmful outputs
- Improved safety through adversarial testing

**Why Read:** Essential for understanding AI safety and alignment challenges.

---

## Emerging Architectures

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces
**Authors:** Gu & Dao (Carnegie Mellon, Princeton)
**Year:** 2023
**Link:** https://arxiv.org/abs/2312.00752

**Key Contributions:**
- Selective state space models for linear-time sequence modeling
- Potential alternative to transformers for long sequences
- Hardware-aware algorithm design
- Competitive performance with better efficiency

**Why Read:** Promising alternative architecture that may address transformer limitations.

---

### Retentive Network: A Successor to Transformer for Large Language Models
**Authors:** Sun et al. (Microsoft, Tsinghua)
**Year:** 2023
**Link:** https://arxiv.org/abs/2307.08621

**Key Contributions:**
- Retention mechanism as alternative to attention
- Linear complexity with competitive performance
- Parallel and recurrent formulations
- Improved training and inference efficiency

**Why Read:** Another promising direction beyond pure transformer architectures.

---

## How to Use This List

**For Beginners:**
Start with:
1. "Attention is All You Need" (foundation)
2. "Language Models are Few-Shot Learners" (GPT-3)
3. "Chain-of-Thought Prompting" (practical technique)

**For Practitioners:**
Focus on:
- LoRA and QLoRA (efficient fine-tuning)
- RAG (practical deployment)
- InstructGPT (alignment)
- FlashAttention (optimization)

**For Researchers:**
Read comprehensively across sections, with special attention to:
- Scaling Laws
- Constitutional AI
- Emerging Architectures
- Evaluation & Safety papers

**For Specific Domains:**
- **Vision:** ViT, CLIP, LLaVA
- **Generation:** DDPM, Stable Diffusion, DALL-E 2
- **Agents:** ReAct, Tree of Thoughts
- **Efficiency:** LoRA, QLoRA, FlashAttention

---

## Additional Resources

- **arXiv:** https://arxiv.org/ - Preprint server for research papers
- **Papers with Code:** https://paperswithcode.com/ - Papers with implementation code
- **Hugging Face Papers:** https://huggingface.co/papers - Daily trending AI papers
- **AI Conference Proceedings:** NeurIPS, ICML, ICLR, ACL, CVPR, EMNLP

---

## Contributing

This list focuses on papers from 2020-2025. For suggestions or updates, consider the impact, citations, and practical applicability of papers.

**Last Updated:** November 2025
