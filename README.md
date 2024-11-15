AI конвеер для обработки текстовых данных.

Умеет обрабатывать файлы в папке, создавать на каждый из файлов ответный файл, на основании знаний в папке RAG и своей роли system_prompt.
Если удалить все промпты папки prompts то программа будет работать в режиме чата.

Системные требования.
    С предложенными моделями тестировалось только на macbook pro m2max 32гб.
    Потребляло 18гб памяти, работало шустро.
    Если компьютер менее мощный можно выбрать модели послабее, на том же сайте.

Установка и настройка.

Тестировал на Python 3.11.6.

    1.Создаём или переходим в любую папку где ведем проекты

    2.Создаём и активируем
        .venv python3 -m venv .venv
        source .venv/bin/activate
    3.Клонируем репу себе и заходим в папку
        git clone https://github.com/Vinttri/AI_ASSIST.git
        cd AI_ASSIST
    4.Ставим зависимости
        pip install -r requirements.txt
    5.Скачать файлы моделей и положить в папку models
        curl -L -o models/nomic-embed-text-v1.5.Q4_0.gguf "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_0.gguf?download=true"
        curl -L -o models/Qwen2.5-Coder-32B-Instruct-Q4_0.gguf "https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q4_0.gguf?download
        Если тут проблемы то нужно сначала зарегистрироваться на сайте https://huggingface.co/
    6.Запускаем
        python ai_assist.py


Настройки.

    У приложения есть файл настроек где можно регулировать как папки хранения файлов, так и настройки моделей. Файл настроек должен быть в той же папки что и файл приложения и называться ai_assist.cfg Все комментарии по настройкам в файле.


Пример работы.(приложены к программе).

    https://github.com/Vinttri/AI_ASSIST/blob/main/Конвейер%20AI%20обработки%20данных.docx

Модели, на которых тестировал(нужны обе):

    https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_0.gguf?download=true

    https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q4_0.gguf?download=true

    По идее будут работать любые LLAMA-подобные модели формата gguf.
