from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer

# -------------------------------
# 1️⃣ Подготовка данных
# -------------------------------
data = [
    {"input": "dog is a", "output": " animal"},
    {"input": "knife is used for", "output": " cutting"},
    {"input": "cat is a", "output": " pet"},
    {"input": "speling eror example", "output": " spelling error example"},
]

dataset = Dataset.from_list(data)

# -------------------------------
# 2️⃣ Токенизатор
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 не имеет pad

MAX_LENGTH = 32

# -------------------------------
# 3️⃣ Функция токенизации
# -------------------------------
def tokenize_fn(example):
    max_length=MAX_LENGTH

    # токенизируем input
    input_ids = tokenizer(
        example["input"],
        truncation=True,
        max_length=max_length - 1,
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    print(input_ids)

    # токенизируем output
    output_ids = tokenizer(
        example["output"],
        truncation=True,
        max_length=max_length - len(input_ids),
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    # labels: игнорируем input, учим только на output
    labels = [-100] * len(input_ids) + output_ids

    # паддинг
    input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
    input_ids = input_ids[:max_length]

    labels = labels + [-100] * (max_length - len(labels))
    labels = labels[:max_length]

    attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_ids]

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return result


tokenized_dataset = dataset.map(tokenize_fn)

# -------------------------------
# 4️⃣ Модель
# -------------------------------
model = AutoModelForCausalLM.from_pretrained("gpt2")

# -------------------------------
# 5️⃣ SFTTrainer
# -------------------------------
training_args = TrainingArguments(
    #output_dir="./sft_gpt2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=200,
    learning_rate=5e-5,
    logging_steps=32,
    save_strategy="no",
    lr_scheduler_type="constant",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    #dataset_text_field="input",
)

# -------------------------------
# 6️⃣ Обучение
# -------------------------------
trainer.train()

# -------------------------------
# 7️⃣ Тест генерации
# -------------------------------
prompt = "dog is a"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
