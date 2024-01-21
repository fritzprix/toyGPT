
# toyGPT - A Hands-On Project in Building a Basic GPT Model

Almost 7 years have passed since the [Transformer paper](https://arxiv.org/abs/1706.03762) was published by Google researchers, marking a remarkable milestone in technology. Today, large language models, epitomized by services like ChatGPT, are ubiquitous in our daily lives. Although I'm no longer an active developer, understanding the basic building blocks of these achievements seems essential for my role as a tech investment manager. Therefore, to enhance my understanding, I've decided to build a small GPT model from scratch as a learning experience.

## Approach - Top Priority in Readability

Creating a production-ready transformer model involves numerous complexities, but my goal is to simplify the process for educational purposes. I aim to design it in such a way that even those new to Python, including my future self, can easily understand its inner workings.


## Model Architecture

| Model Name     | Number of Layers | Number of Attention Heads | Block Size | Embedding Size | Parameter Count (Million) |
|----------------|------------------|---------------------------|------------|----------------|---------------------------|
| toyGPT.small   | 12               | 8                         | 512        | 768            | 123.70                    |
| toyGPT.medium  | 16               | 8                         | 512        | 1024           | 253.06                    |
| toyGPT.large   | 24               | 16                        | 512        | 2048           | 1311.58                   |

## Key Techniques and Tools

- **Weight Tying:** Implemented in the embedding layer to reduce the model's overall parameter count, enhancing efficiency.
- **Flash Attention:** is used as main algorithm for scaled dot product attention for training efficiency, there is also my own implementation from scrach but it's only for learning purpose
- **Absolute Position Embedding:** Inspired by the technique outlined in the "Attention Is All You Need" paper, this embedding method is integral to the model's ability to understand sequence order.

## Tokenization

- **Tokenizer Used:** GPT2Tokenizer from Hugging Face. This choice was made for convenience and to ensure compatibility with widely used standards in NLP.

## Generation

- No Sampling, Just Greedy

## Training and Experimentation Details

For an in-depth look at the training process, experiment setups, and results, visit our project page on Weights & Biases:

[ToyGPT Training Details on Weights & Biases](https://wandb.ai/dwidlee/toygpt/overview?workspace=user-dwidlee)

||[checkpoint](dwidlee/toygpt/model-99cmyt9l:v29)|
|-|-|
|train loss|4.505|
|val. loss|4.548|

### Dataset Sources

- **Wikimedia/Wikisource:** A comprehensive collection of texts from Wikisource, specifically the `20231201.en` subset, providing a wide range of literary and historical texts.
- **Togethercomputer/RedPajama-Data-1T-Sample:** A curated dataset from Togethercomputer, representing a diverse sample of text data.

|split|token count|
|-|-|
|training| 1.500872004 billion|
|validation| 0.01516361 billion|
|test | 0.015308814 billion|

### Note

This project is currently limited to the small model due to compute resource constraints. Plans for expanding to medium and large models are in place, contingent on resource availability.

## Requirements

Before starting, ensure you have the following requirements installed:

- Python 3.6 or higher
- PyTorch
- Transformers library
- Lightning library
- WANDB (optional for logging)

## Training the Model

To train the model from scratch, follow these steps:

1. **Set up your configuration**: Customize your training settings by modifying `config.json`.

2. **Start training**: Run the following command in your terminal:

```shell
python . train --config <path_to_config> --batch <batch_size> --lr <learning_rate> --wd <weight_decay> --precision <precision> --wnb <wandb_logging>
```

or simply

```shell
python . train
```

## Resuming Training

To resume training from the last checkpoint, use the following command:

```shell
python . resume --config <path_to_config> --wnb <wandb_logging> --precision <precision>
```

or simply

```shell
python . resume
```

## Generating Text

For text generation using a trained model, follow these steps:

1. **Prepare a prompt**: Decide on the text prompt you want to use for generation.

2. **Run generation**: Execute the command below:

```shell
python . generate --prompt "Your prompt here" --model <path_to_model>
```

or simply

```shell
python . generate --prompt "Your prompt here"
```

## Some generation examples

```shell
python . generate --prompt "This is the end"
```

> This is the end of the year, and I’m not sure what I’m doing is to be a little more interested in the future. I’m not sure what I’m doing is to be a little more involved in the future. I’m not sure what I’m doing is to be a good part of the world....

```shell
python . generate --prompt "Socrates said that"
```

> Socrates said that the government has not yet been able to make any changes to the government’s budget, but that the government has not yet made any changes to the government’s budget. “The government has not yet made any changes to the government’s budget,” said the government. “The government has not yet made any changes to the government’s budget, but has not yet made any changes to the state’s budget.” The government has not yet made any changes to the state’s budget, but has not yet made any changes to the state’s budget, but has not yet made any changes to the state’s budget,
