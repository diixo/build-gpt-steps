

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer

# 1. Загружаем предобученную GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 не имеет токена padding, нужно добавить
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Подготавливаем данные для SFT
# Формат: список словарей с ключами "input" и "output"
train_data = [
    {"input": "Translate to French: Hello", "output": "Bonjour"},
    {"input": "Translate to French: How are you?", "output": "Comment ça va?"},
    {"input": "ottention", "output": "attention"},
]

# 3. Создаем SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    tokenizer=tokenizer,
    max_seq_length=128,
    batch_size=2,
    learning_rate=5e-5,
    num_epochs=3,
    output_dir="./sft_gpt2"
)

# 4. Обучение
trainer.train()

# 5. Сохранение модели
trainer.save_model("./sft_gpt2")
