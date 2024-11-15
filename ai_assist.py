import os
import numpy as np
import faiss
import logging
import time
import pickle
import torch
import psutil
from llama_cpp import Llama
import configparser

# ==========================
# Load Configuration
# ==========================

# Create a ConfigParser object to read the configuration file
config = configparser.ConfigParser()
config.read('ai_assist.cfg')

# Paths from the configuration file
PROMPTS_DIR = config.get('Paths', 'prompts_dir')
RAGS_DIR = config.get('Paths', 'rags_dir')
ANSWERS_DIR = config.get('Paths', 'answers_dir')
INDEX_FILE = config.get('Paths', 'index_file')
EMBEDDINGS_FILE = config.get('Paths', 'embeddings_file')
EMBEDDING_MODEL_PATH = config.get('Paths', 'embedding_model_path')
MAIN_MODEL_PATH = config.get('Paths', 'main_model_path')
SYSTEM_PROMPT_FILE = config.get('Paths', 'system_prompt_file')

# Text generation settings
MAX_NEW_TOKENS = config.getint('GenerationSettings', 'max_new_tokens')
TEMPERATURE = config.getfloat('GenerationSettings', 'temperature')
TOP_P = config.getfloat('GenerationSettings', 'top_p')
STOP_SYMBOLS = config.get('GenerationSettings', 'stop_symbols').split(',')

# RAG settings
TOP_K = config.getint('RAGSettings', 'top_k')

# Logging settings
LOG_FILE = config.get('Logging', 'log_file')
LOG_LEVEL = getattr(logging, config.get('Logging', 'log_level').upper())
LOG_MEMORY_USAGE = config.getboolean('Logging', 'log_memory_usage')

# Model settings
EMBEDDING_MODEL_N_CTX = config.getint('ModelSettings', 'embedding_model_n_ctx')
EMBEDDING_MODEL_N_BATCH = config.getint('ModelSettings', 'embedding_model_n_batch')
MAIN_MODEL_N_CTX = config.getint('ModelSettings', 'main_model_n_ctx')
MAIN_MODEL_N_BATCH = config.getint('ModelSettings', 'main_model_n_batch')
USE_MLOCK = config.getboolean('ModelSettings', 'use_mlock')
VERBOSE = config.getboolean('ModelSettings', 'verbose')
N_GPU_LAYERS = config.getint('ModelSettings', 'n_gpu_layers')

# ==========================
# Настройка логирования
# ==========================

logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ==========================
# Функция для логирования памяти
# ==========================

def log_memory_usage():
    """Logs the current memory usage of the process."""
    if LOG_MEMORY_USAGE:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.info(f"Память RSS: {mem_info.rss / (1024 ** 2):.2f} МБ")
        logging.info(f"Память VMS: {mem_info.vms / (1024 ** 2):.2f} МБ")

# ==========================
# Инициализация моделей
# ==========================

# Determine the device to use (MPS for Apple Silicon, CUDA for GPUs, or CPU)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

logging.info(f"Используется устройство: {DEVICE}")

# Initialize the embedding model
logging.info(f"Модель для эмбеддингов загружена из: {EMBEDDING_MODEL_PATH}")
embedding_model = Llama(
    model_path=EMBEDDING_MODEL_PATH,
    embedding=True,
    n_threads=os.cpu_count(),
    n_ctx=EMBEDDING_MODEL_N_CTX,
    n_batch=EMBEDDING_MODEL_N_BATCH,
    use_mlock=USE_MLOCK,
    verbose=VERBOSE,
    n_gpu_layers=N_GPU_LAYERS,
)

# Initialize the main model
logging.info(f"Основная модель загружена из: {MAIN_MODEL_PATH}")
main_model = Llama(
    model_path=MAIN_MODEL_PATH,
    n_threads=os.cpu_count(),
    n_ctx=MAIN_MODEL_N_CTX,
    n_batch=MAIN_MODEL_N_BATCH,
    use_mlock=USE_MLOCK,
    verbose=VERBOSE,
    n_gpu_layers=N_GPU_LAYERS,
)

# ==========================
# Начало программы
# ==========================

def main():
    """Main function to run the application."""
    start_time = time.time()
    logging.info("Запуск программы")
    log_memory_usage()

    # Create necessary directories if they don't exist
    for directory in [PROMPTS_DIR, RAGS_DIR, ANSWERS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Создана папка: {directory}")
        else:
            logging.info(f"Папка уже существует: {directory}")

    # Load the system prompt from file
    system_prompt = load_system_prompt()

    # Step 1: Load documents from the RAG directory
    documents = load_documents()
    logging.info(f"Загружено документов: {len(documents)}")
    log_memory_usage()

    # Step 2: Index the documents
    index = index_documents(documents)
    logging.info("Документы проиндексированы")
    log_memory_usage()

    # Step 3: Process prompts
    process_prompts(documents, index, system_prompt)

    end_time = time.time()
    logging.info(f"Завершение программы. Общее время работы: {end_time - start_time:.2f} секунд")
    log_memory_usage()

# Load the system prompt from a file
def load_system_prompt():
    """Loads the system prompt from a file."""
    if not os.path.exists(SYSTEM_PROMPT_FILE):
        # Create an empty system prompt file if it doesn't exist
        os.makedirs(os.path.dirname(SYSTEM_PROMPT_FILE), exist_ok=True)
        with open(SYSTEM_PROMPT_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        logging.info(f"Файл системного промпта создан: {SYSTEM_PROMPT_FILE}")
        return ""
    else:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
            logging.info(f"Системный промпт загружен из: {SYSTEM_PROMPT_FILE}")
            return system_prompt

# Step 1: Load documents from the RAG directory
def load_documents():
    """Loads documents from the RAG directory."""
    documents = []
    for filename in os.listdir(RAGS_DIR):
        filepath = os.path.join(RAGS_DIR, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                logging.debug(f"Загружен документ: {filename}")
    return documents

# Step 2: Index the documents
def index_documents(documents):
    """Indexes the documents using pip ."""
    if documents:
        if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
            # Load the index and embeddings from files
            logging.info("Загружаем сохраненный индекс и эмбеддинги документов")
            with open(EMBEDDINGS_FILE, 'rb') as f:
                document_embeddings = pickle.load(f)
            index = faiss.read_index(INDEX_FILE)
        else:
            # Generate embeddings and create a new index
            logging.info("Начало векторизации документов")
            start_time = time.time()
            document_embeddings = embed_texts(documents)
            end_time = time.time()
            logging.info(f"Векторизация документов завершена за {end_time - start_time:.2f} секунд")

            dimension = document_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(document_embeddings))

            # Save the index and embeddings to files
            faiss.write_index(index, INDEX_FILE)
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(document_embeddings, f)
            logging.info("Индекс и эмбеддинги сохранены в файлы")
        return index
    else:
        logging.warning("Нет документов для индексации")
        return None

# Function to get embeddings for texts
def embed_texts(texts):
    """Generates embeddings for a list of texts."""
    logging.info("Получение эмбеддингов для текстов")
    embeddings = []
    for idx, text in enumerate(texts):
        logging.debug(f"Векторизация текста {idx+1}/{len(texts)}")
        embedding = embedding_model.embed(text)
        embeddings.append(embedding)
        log_memory_usage()
    embeddings = np.array(embeddings)
    logging.debug(f"Получено эмбеддингов: {len(embeddings)}")
    return embeddings

# Step 3: Retrieve relevant text based on the query
def retrieve_relevant_text(query, index, documents, top_k=TOP_K):
    """Retrieves relevant documents for the query."""
    if index is not None:
        logging.info("Извлечение релевантного контекста")
        start_time = time.time()
        query_embedding = embed_texts([query])
        distances, indices = index.search(np.array(query_embedding), min(top_k, len(documents)))
        unique_indices = list(set(indices[0]))
        relevant_texts = [documents[idx] for idx in unique_indices]
        end_time = time.time()
        logging.info(f"Контекст извлечен за {end_time - start_time:.2f} секунд")
        return "\n\n".join(relevant_texts)
    else:
        logging.warning("Индекс не создан, контекст будет пустым")
        return ""

# Step 4: Process prompts from files or user input
def process_prompts(documents, index, system_prompt):
    """Processes prompts and generates answers."""
    # Exclude 'user_system_prompt.txt' from prompt files
    prompt_files = [f for f in os.listdir(PROMPTS_DIR)
                    if os.path.isfile(os.path.join(PROMPTS_DIR, f)) and f != os.path.basename(SYSTEM_PROMPT_FILE)]

    if prompt_files:
        logging.info(f"Обработка файлов в папке prompts: {prompt_files}")
        # Remove existing answer files except those starting with 'user_'
        for filename in os.listdir(ANSWERS_DIR):
            if not filename.startswith('user_'):
                os.remove(os.path.join(ANSWERS_DIR, filename))
                logging.debug(f"Удален файл ответа: {filename}")

        for filename in prompt_files:
            logging.info(f"Обработка файла: {filename}")
            prompt_path = os.path.join(PROMPTS_DIR, filename)
            with open(prompt_path, 'r', encoding='utf-8') as file:
                query = file.read()
                logging.debug(f"Содержимое запроса: {query}")

            # Retrieve relevant context
            context = retrieve_relevant_text(query, index, documents)

            # Generate model's answer
            answer_text = generate_text(query, context, system_prompt)

            # Save the answer to a file in the answers directory
            answer_filename = os.path.splitext(filename)[0] + '_answer.txt'
            answer_path = os.path.join(ANSWERS_DIR, answer_filename)

            with open(answer_path, 'w', encoding='utf-8') as answer_file:
                answer_file.write(answer_text)

            print(f"Ответ для файла {filename} сохранен в {answer_filename}")
            logging.info(f"Ответ сохранен в файл: {answer_filename}")
    else:
        # If no prompt files, accept user input
        logging.info("Нет файлов в prompts, ожидается ввод от пользователя")
        print("Введите ваш вопрос (для выхода введите 'quit' или 'выход'):")
        while True:
            query = input()
            if query.lower() in ['quit', 'выход']:
                print("Выход из программы.")
                logging.info("Пользователь завершил сеанс.")
                break
            logging.debug(f"Вопрос пользователя: {query}")

            # Retrieve relevant context
            context = retrieve_relevant_text(query, index, documents)

            # Generate model's answer
            answer_text = generate_text(query, context, system_prompt)

            # Save the answer to a file 'user_answer.txt' in append mode
            answer_path = os.path.join(ANSWERS_DIR, 'user_answer.txt')

            with open(answer_path, 'a', encoding='utf-8') as answer_file:
                answer_file.write(f"Вопрос: {query}\nОтвет: {answer_text}\n{'-'*40}\n")

            print("Ответ модели:")
            print(answer_text)
            logging.info("Ответ выведен на экран и сохранен в файл user_answer.txt")
            print("\nВведите следующий вопрос (для выхода введите 'quit' или 'выход'):")

# Function to generate text using the main model
def generate_text(query, context, system_prompt):
    """Generates text response from the main model."""
    logging.info("Генерация текста моделью")
    try:
        # Formulate the prompt, including context if available
        if context:
            prompt = f"{system_prompt}\n\nКонтекст:\n{context}\n\nПользователь: {query}\nАссистент:"
        else:
            prompt = f"{system_prompt}\n\nПользователь: {query}\nАссистент:"
        logging.debug(f"Промпт для модели:\n{prompt}")

        # Log the start of generation
        logging.info("Начало генерации ответа моделью")
        log_memory_usage()

        output = main_model(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            echo=False,
            stop=STOP_SYMBOLS
        )
        answer = output['choices'][0]['text'].strip()

        # Log the completion of generation
        logging.info("Генерация ответа завершена")
        logging.debug(f"Ответ модели:\n{answer}")
        log_memory_usage()

        return answer
    except Exception as e:
        logging.error(f"Ошибка при генерации текста: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

if __name__ == '__main__':
    main()
