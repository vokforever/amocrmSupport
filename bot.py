import os
import asyncio
import requests
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import OpenAI
from tavily import TavilyClient
from supabase import create_client, Client
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import aiohttp
from bs4 import BeautifulSoup
import logging
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Проверка наличия необходимых переменных окружения
required_env_vars = [
    "TELEGRAM_BOT_TOKEN",
    "OPENROUTER_API_KEY",
    "TAVILY_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "MISTRAL_API_KEY"
]

for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Переменная окружения {var} не найдена. Пожалуйста, проверьте файл .env")

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация клиентов
bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()
scheduler = AsyncIOScheduler()

# OpenRouter (с правильным именем модели)
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Tavily API
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Supabase
supabase: Client = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

# База знаний amoCRM
AMOCRM_SUPPORT_URL = "https://www.amocrm.ru/support"

# Состояния для FSM
class SupportStates(StatesGroup):
    waiting_for_feedback = State()
    waiting_for_clarification = State()

# Функция для создания клавиатуры обратной связи
def get_feedback_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, помогло",
        callback_data="feedback_yes"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, не помогло",
        callback_data="feedback_no"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🔍 Поищи еще",
        callback_data="search_more"
    ))
    builder.adjust(2, 1)  # Первая строка с 2 кнопками, вторая с 1
    return builder.as_markup()

# Функция для создания клавиатуры уточнения
def get_clarification_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="🔍 Уточнить вопрос",
        callback_data="clarify_question"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📞 Связаться с поддержкой",
        callback_data="contact_support"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🔄 Попробовать еще раз",
        callback_data="try_again"
    ))
    builder.adjust(1)
    return builder.as_markup()

# Функция для получения эмбеддинга с помощью Mistral AI
def get_embedding(text: str) -> List[float]:
    """Получение эмбеддинга текста с помощью Mistral AI"""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-embed",
            "input": text
        }
        response = requests.post("https://api.mistral.ai/v1/embeddings", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        logging.error(f"Ошибка при получении эмбеддинга: {e}")
        return []

# Функция для косинусного сходства
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Вычисление косинусного сходства между двумя векторами"""
    try:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except:
        return 0.0

# Функция для векторного поиска
def vector_search(query: str, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Поиск похожих вопросов в векторной базе знаний"""
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        # Получаем все записи с эмбеддингами
        response = supabase.table("knowledge_base_vector").select("*").execute()
        
        results = []
        for item in response.data:
            if item.get("embedding"):
                similarity = cosine_similarity(query_embedding, item["embedding"])
                if similarity >= threshold:
                    results.append((item["question"], item["answer"], similarity))
        
        # Сортируем по схожести
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:3]  # Возвращаем топ-3 результата
    except Exception as e:
        logging.error(f"Ошибка при векторном поиске: {e}")
        return []

# Функция для сохранения в векторную базу знаний
def save_to_vector_knowledge_base(question: str, answer: str, source: str = ""):
    """Сохранение вопроса и ответа с эмбеддингом"""
    try:
        embedding = get_embedding(question)
        if embedding:
            supabase.table("knowledge_base_vector").insert({
                "question": question,
                "answer": answer,
                "source": source,
                "embedding": embedding,
                "created_at": datetime.now().isoformat()
            }).execute()
    except Exception as e:
        logging.error(f"Ошибка при сохранении в векторную базу знаний: {e}")

# Функция для прямого поиска в базе знаний amoCRM
async def search_amocrm_support(query: str) -> str:
    """Поиск в базе знаний amoCRM через их сайт поддержки"""
    try:
        # Используем Tavily для поиска именно в разделе поддержки
        response = tavily_client.search(
            query=f"{query} site:amocrm.ru/support",
            search_depth="advanced",
            max_results=3
        )
        
        results = []
        for result in response["results"]:
            # Дополнительно парсим страницу для получения полного контента
            content = await scrape_amocrm_page(result["url"])
            if content:
                results.append(f"Источник: {result['url']}\n{content}")
            else:
                results.append(f"Источник: {result['url']}\n{result['content']}")
        
        return "\n\n".join(results)
    except Exception as e:
        logging.error(f"Ошибка при поиске в базе знаний amoCRM: {e}")
        return ""

# Функция для парсинга страницы amoCRM
async def scrape_amocrm_page(url: str) -> str:
    """Парсинг конкретной страницы базы знаний amoCRM"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Удаляем ненужные элементы
                    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                        element.decompose()
                    
                    # Ищем основной контент по приоритету селекторов
                    content_selectors = [
                        'article', 
                        '.content', 
                        '.article-content',
                        '.support-content',
                        'main',
                        '.main-content'
                    ]
                    
                    content = None
                    for selector in content_selectors:
                        content = soup.select_one(selector)
                        if content:
                            break
                    
                    if content:
                        # Извлекаем текст и очищаем его
                        text = content.get_text(separator='\n', strip=True)
                        # Удаляем лишние пробелы и пустые строки
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        cleaned_text = '\n'.join(lines)
                        # Ограничиваем длину для экономии токенов
                        return cleaned_text[:2000] + "..." if len(cleaned_text) > 2000 else cleaned_text
        return ""
    except Exception as e:
        logging.error(f"Ошибка при парсинге страницы {url}: {e}")
        return ""

# Функция для поиска в своей базе знаний (Supabase)
def search_knowledge_base(query: str) -> str:
    """Поиск в собственной базе знаний"""
    try:
        # Сначала пробуем векторный поиск
        vector_results = vector_search(query)
        if vector_results:
            return "\n\n".join([f"Вопрос: {q}\nОтвет: {a}" for q, a, _ in vector_results])
        
        # Если векторный поиск не дал результатов, используем текстовый поиск
        response = supabase.table("knowledge_base").select("*").execute()
        results = [item["answer"] for item in response.data if query.lower() in item["question"].lower()]
        return "\n".join(results) if results else ""
    except Exception as e:
        logging.error(f"Ошибка при поиске в базе знаний: {e}")
        return ""

# Функция для сохранения успешного ответа в базу знаний
def save_to_knowledge_base(question: str, answer: str, source: str = ""):
    """Сохранение успешного ответа в базу знаний"""
    try:
        supabase.table("knowledge_base").insert({
            "question": question, 
            "answer": answer,
            "source": source,
            "created_at": datetime.now().isoformat()
        }).execute()
        
        # Также сохраняем в векторную базу знаний
        save_to_vector_knowledge_base(question, answer, source)
    except Exception as e:
        logging.error(f"Ошибка при сохранении в базу знаний: {e}")

# Функция для сохранения обратной связи пользователя
def save_user_feedback(user_id: int, question: str, helped: bool):
    """Сохранение обратной связи пользователя"""
    try:
        supabase.table("user_feedback").insert({
            "user_id": user_id,
            "question": question,
            "helped": helped,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logging.error(f"Ошибка при сохранении обратной связи: {e}")

# Функция для генерации ответа с помощью OpenRouter (исправленная)
async def generate_answer(question: str, context: str = "") -> str:
    """Генерация ответа с помощью OpenRouter"""
    models_to_try = [
        "openrouter/horizon-beta",      # Основная модель
        "openai/gpt-oss-20b:free",     # Запасной вариант 1 (бесплатная модель)
        "mistralai/mistral-large-2407", # Запасной вариант 2
        "mistralai/mistral-7b-instruct" # Запасной вариант 3
    ]
    
    last_error = None
    
    for model in models_to_try:
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Ты — эксперт технической поддержки по amoCRM (Kommo). 
                    Твоя задача — помогать менеджерам продаж недвижимости решать проблемы с amoCRM.
                    Отвечай максимально релевантно, используя предоставленный контекст.
                    Если в контексте есть точный ответ из официальной базы знаний — используй его.
                    Всегда указывай прямую ссылку на источник, если она есть в контексте.
                    Отвечай на русском языке.
                    Структурируй ответ с использованием эмодзи для лучшего восприятия."""
                }
            ]
            
            if context:
                messages.append({"role": "system", "content": f"Информация из базы знаний amoCRM:\n{context}"})
            
            messages.append({"role": "user", "content": question})
            
            # Добавляем заголовки для OpenRouter
            extra_headers = {
                "HTTP-Referer": "https://github.com/vokforever/amocrm-support",  # Замените на ваш URL
                "X-Title": "amoCRM Support Bot"  # Замените на название вашего бота
            }
            
            completion = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                extra_headers=extra_headers
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            last_error = e
            logging.warning(f"Ошибка при использовании модели {model}: {e}")
            continue
    
    # Если все модели не сработали
    logging.error(f"Все модели недоступны. Последняя ошибка: {last_error}")
    return "😔 К сожалению, произошла ошибка при генерации ответа. Все модели временно недоступны. Попробуйте повторить запрос позже."

# Функция для поиска в интернете через Tavily
async def search_web(query: str) -> str:
    """Поиск в интернете"""
    try:
        response = tavily_client.search(query, max_results=3)
        return "\n".join([f"{result['content']}\nИсточник: {result['url']}" for result in response["results"]])
    except Exception as e:
        logging.error(f"Ошибка при поиске в интернете: {e}")
        return ""

# Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: types.Message):
    await message.answer(
        "👋 Привет! Я бот технической поддержки по amoCRM для менеджеров продаж недвижимости.\n\n"
        "📚 Я использую официальную базу знаний amoCRM: https://www.amocrm.ru/support\n\n"
        "💡 Просто задайте ваш вопрос, и я найду для вас ответ!\n\n"
        "📊 Доступные команды:\n"
        "/stats - моя статистика помощи\n"
        "/history - история обращений\n"
        "/clear - очистить историю"
    )

# Обработчик команды /stats
@dp.message(Command("stats"))
async def stats_command(message: types.Message):
    try:
        response = supabase.table("user_feedback").select("*").eq("user_id", message.from_user.id).execute()
        total = len(response.data)
        helped = sum(1 for item in response.data if item["helped"])
        
        await message.answer(
            f"📊 Ваша статистика:\n"
            f"Всего вопросов: {total}\n"
            f"Помогло ответов: {helped}\n"
            f"Успешность: {helped/total*100:.1f}%" if total > 0 else "📊 У вас пока нет статистики"
        )
    except Exception as e:
        logging.error(f"Ошибка при получении статистики: {e}")
        await message.answer("😔 Не удалось загрузить статистику")

# Обработчик команды /history
@dp.message(Command("history"))
async def history_command(message: types.Message):
    try:
        response = supabase.table("user_feedback").select("*").eq("user_id", message.from_user.id).order("created_at", desc=True).limit(5).execute()
        
        if response.data:
            history_text = "📝 Последние вопросы:\n\n"
            for item in response.data:
                status = "✅" if item["helped"] else "❌"
                history_text += f"{status} {item['question'][:50]}...\n"
            
            await message.answer(history_text)
        else:
            await message.answer("📝 У вас пока нет истории обращений")
    except Exception as e:
        logging.error(f"Ошибка при получении истории: {e}")
        await message.answer("😔 Не удалось загрузить историю")

# Обработчик команды /clear
@dp.message(Command("clear"))
async def clear_command(message: types.Message):
    try:
        supabase.table("user_feedback").delete().eq("user_id", message.from_user.id).execute()
        await message.answer("🗑️ Ваша история очищена")
    except Exception as e:
        logging.error(f"Ошибка при очистке истории: {e}")
        await message.answer("😔 Не удалось очистить историю")

# Основной обработчик сообщений
@dp.message(F.text)
async def handle_message(message: types.Message, state: FSMContext):
    question = message.text
    chat_id = message.chat.id
    user_id = message.from_user.id
    
    # Отправляем сообщение о начале поиска
    processing_msg = await message.answer("🔍 Ищу ответ в официальной базе знаний amoCRM...")
    
    # 1. Сначала ищем в официальной базе знаний amoCRM
    amocrm_context = await search_amocrm_support(question)
    
    if amocrm_context:
        await processing_msg.edit_text("📚 Найдено в базе знаний amoCRM. Генерирую ответ...")
        answer = await generate_answer(question, amocrm_context)
        source = "официальной базы знаний amoCRM"
    else:
        # 2. Если не нашли в базе знаний, ищем в своей базе знаний
        await processing_msg.edit_text("🗂️ Ищу в накопленной базе знаний...")
        kb_context = search_knowledge_base(question)
        
        if kb_context:
            await processing_msg.edit_text("💡 Найдено в базе знаний. Генерирую ответ...")
            answer = await generate_answer(question, kb_context)
            source = "накопленной базы знаний"
        else:
            # 3. Если нигде не нашли, ищем в интернете
            await processing_msg.edit_text("🌐 Ищу дополнительную информацию в интернете...")
            web_context = await search_web(f"{question} amoCRM помощь")
            answer = await generate_answer(question, web_context)
            source = "интернета"
    
    await processing_msg.delete()
    
    # Отправляем ответ с указанием источника
    await message.answer(f"{answer}\n\n📖 *Источник: {source}*", parse_mode="Markdown")
    await message.answer("❓ Помог ли вам мой ответ?", reply_markup=get_feedback_keyboard())
    
    # Установка состояния ожидания обратной связи
    await state.set_state(SupportStates.waiting_for_feedback)
    await state.update_data(
        question=question,
        answer=answer,
        source=source,
        attempts=0,
        user_id=user_id
    )
    
    # Планируем напоминание через час
    scheduler.add_job(
        send_reminder,
        "date",
        run_date=datetime.now() + timedelta(hours=1),
        args=[chat_id],
        id=f"reminder_{chat_id}"
    )

# Обработчик кнопок обратной связи
@dp.callback_query(F.data.in_(["feedback_yes", "feedback_no", "search_more"]))
async def handle_feedback_callback(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    question = data["question"]
    answer = data["answer"]
    source = data["source"]
    attempts = data.get("attempts", 0)
    user_id = data.get("user_id", callback.from_user.id)
    chat_id = callback.message.chat.id
    
    # Удаляем запланированное напоминание
    try:
        scheduler.remove_job(f"reminder_{chat_id}")
    except:
        pass
    
    if callback.data == "feedback_yes":
        # Сохранение успешного ответа в базу знаний (только если не из официальной документации)
        if source != "официальной базы знаний amoCRM":
            save_to_knowledge_base(question, answer, source)
        
        # Сохраняем обратную связь
        save_user_feedback(user_id, question, True)
        
        await callback.message.edit_text(
            "✅ Отлично! Я рад, что смог помочь.\n"
            "Если у вас появятся еще вопросы по amoCRM — обращайтесь! 😊"
        )
        await state.clear()
        
    elif callback.data == "feedback_no":
        if attempts < 2:  # Максимум 2 дополнительные попытки
            await callback.message.edit_text(
                "😔 Понимаю, что ответ не помог. Что вы хотите сделать дальше?",
                reply_markup=get_clarification_keyboard()
            )
            # Обновляем состояние
            await state.update_data(attempts=attempts + 1)
        else:
            # Сохраняем обратную связь
            save_user_feedback(user_id, question, False)
            
            await callback.message.edit_text(
                "😔 К сожалению, я не смог найти достаточно информации по вашему вопросу.\n\n"
                "Рекомендую:\n"
                "• 🔍 Попробуйте поискать самостоятельно в базе знаний: https://www.amocrm.ru/support\n"
                "• 📞 Обратиться в официальную поддержку amoCRM\n"
                "• 💬 Проверить раздел помощи в вашем аккаунте amoCRM"
            )
            await state.clear()
    
    elif callback.data == "search_more":
        # Показываем сообщение о поиске
        await callback.message.edit_text("🔍 Ищу дополнительную информацию в интернете...")
        
        # Ищем в интернете с расширенным запросом
        web_context = await search_web(f"{question} amoCRM решение проблемы инструкция")
        new_answer = await generate_answer(question, web_context)
        
        # Отправляем новый ответ
        await callback.message.edit_text(
            f"{new_answer}\n\n📖 *Источник: дополнительный поиск в интернете*",
            parse_mode="Markdown"
        )
        
        # Снова спрашиваем, помог ли ответ
        await bot.send_message(
            chat_id,
            "❓ Помог ли вам этот новый ответ?",
            reply_markup=get_feedback_keyboard()
        )
        
        # Обновляем состояние
        await state.update_data(
            answer=new_answer,
            source="интернета (дополнительный поиск)",
            attempts=attempts + 1
        )

# Обработчик кнопок уточнения
@dp.callback_query(F.data.in_(["clarify_question", "contact_support", "try_again"]))
async def handle_clarification_callback(callback: types.CallbackQuery, state: FSMContext):
    if callback.data == "clarify_question":
        await callback.message.edit_text(
            "🔄 Пожалуйста, уточните ваш вопрос или опишите проблему более подробно."
        )
        await state.set_state(SupportStates.waiting_for_clarification)
        
    elif callback.data == "contact_support":
        await callback.message.edit_text(
            "📞 Для связи с официальной поддержкой amoCRM:\n\n"
            "• Email: support@amocrm.ru\n"
            "• Телефон: 8 (800) 555-36-53\n"
            "• Чат в аккаунте amoCRM\n\n"
            "🔗 Также можете написать в базу знаний: https://www.amocrm.ru/support"
        )
        await state.clear()
        
    elif callback.data == "try_again":
        data = await state.get_data()
        question = data["question"]
        
        await callback.message.edit_text("🔄 Пробую найти другой ответ...")
        
        # Ищем в интернете с другим запросом
        web_context = await search_web(f"{question} amoCRM решение проблемы")
        new_answer = await generate_answer(question, web_context)
        
        await callback.message.edit_text(
            f"{new_answer}\n\n📖 *Источник: дополнительный поиск в интернете*",
            parse_mode="Markdown",
            reply_markup=get_feedback_keyboard()
        )
        
        # Обновляем состояние
        await state.update_data(answer=new_answer, source="интернета")
        await state.set_state(SupportStates.waiting_for_feedback)

# Обработчик уточненного вопроса
@dp.message(SupportStates.waiting_for_clarification)
async def handle_clarification(message: types.Message, state: FSMContext):
    # Обрабатываем уточненный вопрос как новый
    await state.clear()
    await handle_message(message, state)

# Функция для отложенного напоминания
async def send_reminder(chat_id: int):
    try:
        await bot.send_message(
            chat_id,
            "🔔 Напоминаю: помог ли вам мой предыдущий ответ по amoCRM?",
            reply_markup=get_feedback_keyboard()
        )
    except Exception as e:
        logging.error(f"Ошибка при отправке напоминания: {e}")

# Планировщик для отложенных напоминаний
@dp.startup()
async def on_startup():
    scheduler.start()

@dp.shutdown()
async def on_shutdown():
    scheduler.shutdown()

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
