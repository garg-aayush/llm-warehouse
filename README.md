# llm-notebooks


## Blogs
* [Huggingface Blog about efficient training on a single GPU](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one)
* [Padding large-language-models](https://medium.com/towards-data-science/padding-large-language-models-examples-with-llama-2-199fb10df8ff)
* [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
* [Llama.cpp Tutorial](https://www.datacamp.com/tutorial/llama-cpp-tutorial)
* [Introduction to Llama](https://www.datacamp.com/blog/introduction-to-meta-ai-llama)
* [Getting Started with Axolotl for Fine-Tuning LLMs](https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html)
* [The Novoice's LLM Training Guide](https://rentry.org/llm-training#low-rank-adaptation-lora_1)
* [Transformers from scratch](https://benjaminwarner.dev/2023/07/01/attention-mechanism)
* [SFT Training Example using TRL](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb)
* [Perplexity evaluation metric](https://huggingface.co/docs/transformers/en/perplexity)
* [Finetuning LLMs with LoRA and QLoRA](https://lightning.ai/pages/community/lora-insights/) by Sebastian Raschka
* [Zephyr 7B Guide](https://www.kdnuggets.com/exploring-the-zephyr-7b-a-comprehensive-guide-to-the-latest-large-language-model)
* [W&B's Training LLM](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2) by Thomas Capelle
* [Pytorch blog on finetuning LLM](https://pytorch.org/blog/finetune-llms/)

## Notebooks and implementations
* [Minimalistic implementation of LoRA with guidelines](https://colab.research.google.com/drive/1QG1ONI3PfxCO2Zcs8eiZmsDbWPl4SftZ)
* [Maxime Labonne's Fine-tune Llama 2 on Google Colab.ipynb](https://colab.research.google.com/drive/1p68M5E5fZ7kSa7nA-e-20489nuFSXVp2?usp=sharing)
* [Llama from scratch](https://blog.briankitano.com/llama-from-scratch/) by Brian Kitano

## Video Lectures
- [Fine-Tune Llama2 | Step by Step Guide to Customizing Your Own LLM](https://www.youtube.com/watch?v=Pb_RGAl75VE): Great Short introduction on custom SFT data creation and SFT training using TRL
- [Learn RAG from scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8): Learn how to implement RAG (Retrieval Augmented Generation) from scratch using Langchains
- [Aligning LLMs with Direct Preference Optimization](https://www.youtube.com/watch?v=QXVCqtAZAn4&t=2828s)


## Leaderboards
- [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)


## LLM Courses
- [Maxime Labonne LLM course](https://github.com/mlabonne/llm-course?tab=readme-ov-file)
- [W&B Training and Fine-tuning Large Language Models (LLMs)](https://www.wandb.courses/courses/training-fine-tuning-LLMs)
- [W&B Building LLM-Powered Apps](https://www.wandb.courses/courses/building-llm-powered-apps)
- [LLM Datahub](https://github.com/Zjh-819/LLMDataHub) contain datasets for LLM Training


## Github
* [LLaMA model (and others) inference in pure C/C++](https://github.com/ggerganov/llama.cpp)
* [Karpathy's llama C implementation](https://github.com/karpathy/llama2.c?tab=readme-ov-file)
* [LLM evaluation using colabl notebook (AutoEval)](https://github.com/mlabonne/llm-autoeval)
* [Llama recipes by Meta](https://github.com/meta-llama/llama-recipes)
* [Huggingface Alignment Handbook](https://github.com/huggingface/alignment-handbook)
* [LLM foundry by Databricks](https://github.com/mosaicml/llm-foundry?tab=readme-ov-file)

## Datasets
1. [Open-Platypus]([garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)): SFT dataset for improving LLM logical reasoning skills and was used to train the Platypus2 models.
2. [guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k): 1000 example subset of `timdettmers/openassistant-guanaco` dataset in Llama 2's prompt format. Good for learning purposes.
3. [Ultrachat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k):  Filtered version of the UltraChat dataset used to train Zephyr-7B-β model
4. [HF's datasets collections](https://huggingface.co/collections/HuggingFaceH4/awesome-sft-datasets-65788b571bf8e371c4e4241a): Curated SFT datasets.