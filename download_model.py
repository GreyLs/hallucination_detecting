import os
import shutil

model_name = "DeepPavlov/rubert-base-cased"
save_directory = "./saved_models/best_model_Savvy_24"

# Создаем папки, если их нет
if not os.path.exists("model"):
    os.makedirs("model")

if not os.path.exists("tokenizer"):
    os.makedirs("tokenizer")

tokenizer = AutoTokenizer.from_pretrained('./tokenizer')  # Предполагается, что файлы уже находятся в папке

# Сохраняем токенизатор в переменную
loaded_tokenizer = tokenizer

# Проверка наличия файла модели перед копированием
if os.path.exists("saved_models/best_model_Savvy_24/best_model_Savvy_24.pth"):
    shutil.copyfile("saved_models/best_model_Savvy_24/best_model_Savvy_24.pth", "model/best_model_Savvy_24.pth")
else:
    print("Model file not found!")

# Проверка наличия конфигурационного файла перед копированием
if os.path.exists("saved_models/config.json"):
    shutil.copyfile("saved_models/config.json", "model/config.json")
else:
    print("Config file not found!")
