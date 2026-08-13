# Llama 3.2 1B Instruct --- Parameter-Efficient Fine-Tuning with LoRA

A hands-on implementation of **supervised instruction fine-tuning
(SFT)** for `meta-llama/Llama-3.2-1B-Instruct` using **Low-Rank
Adaptation (LoRA)** and the Hugging Face training stack.

Rather than updating all \~1.24B model parameters, this project freezes
the pretrained Llama backbone and trains only **3,407,872 parameters
(\~0.275%)** through LoRA adapters. The final run reached a **training
loss of 0.173438** and **validation loss of 0.173505**.

> **Core idea:** adapt a pretrained instruction-following LLM to a
> structured robot/game QA task while updating less than 0.3% of its
> parameters.

## Results at a Glance

  Metric                                                  Result
  -------------------------------------- -----------------------
  Base model                               Llama 3.2 1B Instruct
  Total parameters after LoRA wrapping             1,239,222,272
  Trainable parameters                                 3,407,872
  Trainable percentage                                   0.2750%
  LoRA rank (`r`)                                             16
  LoRA alpha                                                  32
  LoRA dropout                                              0.05
  Final training step                                      6,106
  Final training loss                               **0.173438**
  Validation loss                                   **0.173505**
  Dataset rows before splitting                           54,269

![Training and Validation Loss](results/training_validation_loss.png)

## End-to-End Pipeline

``` text
Llama 3.2 1B Instruct
        ↓
Dataset loading (54,269 instruction/input/output examples)
        ↓
Train / validation split
        ↓
Structured context + natural-language question + answer
        ↓
Llama role-based chat formatting
        ↓
tokenizer.apply_chat_template()
        ↓
Tokenization
        ↓
LoRA adapter injection into attention projections
        ↓
Causal language-model training
        ↓
Held-out validation
        ↓
Loss analysis + visualization
        ↓
Autoregressive generation
```

## 1. Base Model

``` python
model_name = "meta-llama/Llama-3.2-1B-Instruct"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)
```

The model is a **causal language model**: it learns autoregressively by
predicting the next token from the preceding context.

## 2. Architecture Inspection

Before fine-tuning, the loaded model was inspected directly:

``` python
print("Attention heads:", model.config.num_attention_heads)
print("KV heads:", model.config.num_key_value_heads)
print("Hidden size:", model.config.hidden_size)
print("Layers:", model.config.num_hidden_layers)
```

This connects Llama architecture concepts to the actual implementation
rather than treating the pretrained model as a black box.

## 3. Dataset

The dataset contains **54,269 examples** with:

``` text
instruction
input
output
```

A representative example follows this pattern:

``` text
instruction:
<Result><collected_unsorted_cakes="1" ... /></Result>

input:
How many cakes did you collect and sort?

output:
I collected and sorted 1 cake.
```

The structured result supplies task state, the input asks a
natural-language question about that state, and the output is the
desired assistant response.

The data is split into training and held-out validation subsets before
fine-tuning.

## 4. Chat Formatting

Because the base model is instruction-tuned, each example is represented
as a conversation:

``` python
messages = [
    {
        "role": "user",
        "content": structured_context + "\n\n" + question
    },
    {
        "role": "assistant",
        "content": expected_answer
    }
]
```

The tokenizer's native Llama chat template then converts these roles
into the special-token sequence expected by the model:

``` python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
```

This preserves user/assistant boundaries and end-of-turn markers used by
Llama's instruction format.

## 5. Tokenization and Causal-LM Training

Formatted conversations are tokenized into integer token IDs before
being passed through the Transformer.

``` text
conversation
    ↓
Llama tokenizer
    ↓
input_ids
    ↓
Transformer
    ↓
next-token logits
    ↓
causal LM loss
```

This project performs **causal language modeling**, not masked language
modeling.

## 6. LoRA Configuration

``` python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### Rank --- `r=16`

Instead of learning a dense update to a large weight matrix, LoRA learns
two smaller matrices whose product approximates the update.

``` text
ΔW ≈ B @ A
       ↑
    rank 16
```

Rank controls adapter capacity: larger rank provides more trainable
capacity at the cost of additional parameters and compute.

### Alpha --- `lora_alpha=32`

Alpha controls the scale of the LoRA update. With the conventional
`alpha/r` scaling:

``` text
32 / 16 = 2
```

### Dropout --- `lora_dropout=0.05`

A 5% dropout is applied on the LoRA path during training as
regularization. It is disabled during evaluation/inference.

## 7. Attention Projections Adapted by LoRA

The saved adapter parameters show LoRA A/B matrices attached across
Llama self-attention projections:

``` text
q_proj
k_proj
v_proj
o_proj
```

Examples:

``` text
...self_attn.q_proj.lora_A.default.weight
...self_attn.q_proj.lora_B.default.weight
...self_attn.k_proj.lora_A.default.weight
...self_attn.k_proj.lora_B.default.weight
...self_attn.v_proj.lora_A.default.weight
...self_attn.v_proj.lora_B.default.weight
...self_attn.o_proj.lora_A.default.weight
...self_attn.o_proj.lora_B.default.weight
```

The pretrained projection weights remain frozen; the low-rank matrices
learn the task-specific update.

## 8. Applying PEFT

``` python
from peft import get_peft_model

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Observed:

``` text
trainable params: 3,407,872
all params:       1,239,222,272
trainable%:       0.2750
```

So approximately **99.725% of the parameters are not trained**.

## 9. What Happens During Training

For each batch:

1.  Tokenized conversations enter Llama.
2.  The model performs a forward pass.
3.  Causal-LM loss measures next-token prediction error.
4.  Backpropagation computes gradients.
5.  The optimizer updates the LoRA parameters.
6.  Frozen base-model weights remain unchanged.
7.  The process repeats across training steps.

Representative training logs:

    Step   Training Loss
  ------ ---------------
      50        0.508511
     100        0.394603
     150        0.230713
     200        0.192598
     250        0.192393
     ...             ...
    5900        0.174110
    5950        0.174434
    6000        0.174945
    6050        0.176154
    6100        0.173438

The loss decreases sharply early in training and then stabilizes near
\~0.17--0.19.

## 10. Held-Out Evaluation

Final evaluation:

``` python
{"eval_loss": 0.17350518703460693}
```

  Metric                          Loss
  --------------------- --------------
  Final training loss     **0.173438**
  Validation loss         **0.173505**

The values are very close at the final evaluation point, so there is no
obvious train-validation loss gap in this run.

**Validation loss is not QA accuracy.** This repository intentionally
reports the metric that was actually measured rather than presenting an
unmeasured accuracy value.

## 11. Training Visualization

Trainer logs were converted into a DataFrame:

``` python
import pandas as pd

loss_df = pd.DataFrame([
    {
        "Step": log["step"],
        "Training Loss": log["loss"]
    }
    for log in trainer.state.log_history
    if "loss" in log
])
```

The plot shows the complete training-loss trajectory and the final
held-out validation point.

![Training and Validation Loss](results/training_validation_loss.png)

## 12. Inference

At inference time, a user-only conversation is created and the
generation prompt is appended:

``` python
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50
)
```

Only tokens generated after the prompt are decoded:

``` python
input_length = inputs["input_ids"].shape[-1]

response = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True
)
```

## 13. Qualitative Generation Result

``` text
Question:
How many cakes did you collect and sort?

Expected:
I collected and sorted 5 cakes.

Fine-tuned Llama:
I collected and sorted 5 cakes. The remaining 1 cake was collected but not sorted...
```

The model begins with the correct target answer but continues generating
extra text. This is reported as an actual limitation rather than hiding
it.

The result suggests successful task adaptation on the demonstrated
example while highlighting a future improvement area: **generation
stopping and response conciseness**.

## Technical Stack

-   Python 3.12
-   PyTorch
-   Hugging Face Transformers
-   Hugging Face Datasets
-   PEFT
-   Accelerate
-   Pandas
-   Matplotlib
-   Meta Llama 3.2 1B Instruct

## Repository Structure

``` text
llama3-robotics-lora/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── llama3_lora_finetuning.ipynb
└── results/
    └── training_validation_loss.png
```

## What This Project Demonstrates

-   Loading and inspecting a modern pretrained causal LLM
-   Preparing structured instruction-response data
-   Converting examples into Llama-compatible conversations
-   Applying the model's native chat template
-   Tokenizing data for causal language modeling
-   Configuring PEFT/LoRA
-   Injecting trainable low-rank matrices into attention projections
-   Freezing the pretrained backbone
-   Training only **0.275%** of model parameters
-   Tracking optimization loss over thousands of steps
-   Evaluating on held-out data
-   Visualizing convergence
-   Running autoregressive inference after fine-tuning
-   Analyzing generation limitations instead of reporting only
    successful outputs

## Key Takeaway

``` text
1,239,222,272 total parameters
             ↓
       LoRA adaptation
             ↓
3,407,872 trainable parameters
             ↓
          0.275%
```

The experiment demonstrates that a **1.2B+ parameter instruction model
can be adapted to a specialized structured QA task while training only
3.4M parameters**.

Final measured results:

``` text
Training loss   : 0.173438
Validation loss : 0.173505
```

## Future Work

-   Evaluate exact-match / normalized QA performance across the held-out
    set
-   Compare base-model and fine-tuned-model generations on identical
    prompts
-   Improve end-of-turn stopping and response conciseness
-   Explore 4-bit QLoRA
-   Extend PEFT fine-tuning to multimodal foundation models

## License

The repository's code is governed by the license selected for this
repository. The underlying Llama model and any external dataset remain
subject to their own licenses and terms.
