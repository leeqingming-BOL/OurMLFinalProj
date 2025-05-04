# Machine Translation: Sequence to Sequence Model vs Qwen2.5-3B

This experiment investigates the performance of three models on machine translation tasks: a standard Seq2Seq model, the Qwen-2.5 model, and the Qwen-2.5 model fine-tuned using LoRA.

## How to Run

Using the Seq2Seq model for English-to-Chinese translation tasks.

```
python seq2seq.py
```

Using the Qwen model for English-to-Chinese translation tasks.

1. Download Qwen2.5-3B model

```
git clone https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct
```

2. Install LLaMA-Factory

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

3. To merge LoRA module with Qwen2.5-3B model

To facilitate usage, the pre-trained LoRA module has been uploaded to my GitHub repository.

```
git clone https://github.com/20110728/lora.git
```

After downloading the LoRA module to the local environment, it is integrated with the base Qwen model to enable fine-tuned inference.

```
llamafactory-cli export ../merge_qwen_lora.yaml
```

4. Inference using the original Qwen-2.5 model.

```
llamafactory-cli webchat ../chat_qwen.yaml
```

or

```
llamafactory-cli chat ../chat_qwen.yaml
```

5. Inference using the LoRA fine-tuned Qwen-2.5 model.

```
llamafactory-cli webchat ../chat_qwen_lora.yaml
```

or

```
llamafactory-cli chat ../chat_qwen_lora.yaml
```
