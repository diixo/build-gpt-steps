import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer


# -------------------------------
# 1️⃣ Preparing the data
# -------------------------------
data = [
    {"input": "Aithetic is a", "output": " company."},
    {"input": "Aithetic is a ", "output": "company."},
    #{"input": "Define Aithetic: ", "output": "research company."},
    {"input": "What is Aithetic?", "output": " company."},
]

dataset = Dataset.from_list(data)

# -------------------------------
# 2️⃣ Tokenizer
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 32

# -------------------------------
# 3️⃣ Tokenization
# -------------------------------
def tokenize_fn(example):
    max_length=MAX_LENGTH

    # tokenizes input
    input_ids = tokenizer(
        example["input"],
        truncation=True,
        max_length=max_length - 1,
        add_special_tokens=False,
    )["input_ids"]

    # use only clear input without eos
    attention_mask = [1]*len(input_ids) + [0]*(max_length-len(input_ids))

    input_ids += [tokenizer.eos_token_id]

    # tokenizes output
    output_ids = tokenizer(
        example["output"],
        truncation=True,
        max_length=max_length - len(input_ids),
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    # labels: ignore input, train only using output
    labels = [-100] * len(input_ids) + output_ids

    # Padding
    input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
    input_ids = input_ids[:max_length]

    labels = labels + [-100] * (max_length - len(labels))
    labels = labels[:max_length]

    test_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]
    assert all(x == y for x, y in zip(test_mask, attention_mask))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


dataset = dataset.map(tokenize_fn, batched=False)

# -------------------------------
# 4️⃣ Model
# -------------------------------
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 5️⃣ SFTTrainer
# -------------------------------
training_args = TrainingArguments(
    #output_dir="./sft_gpt2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # gradient on whole batch
    num_train_epochs=100,
    learning_rate=3e-5,
    logging_steps=32,
    save_strategy="no",
    lr_scheduler_type="constant",
    max_grad_norm=1.0,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    #dataset_text_field="input",
)

# -------------------------------
# 6️⃣ Training
# -------------------------------
trainer.train()

# -------------------------------
# 7️⃣ Test of generation
# -------------------------------
prompt = "Aithetic is a"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=2,   # 1 or 2 words with maximal probability
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
