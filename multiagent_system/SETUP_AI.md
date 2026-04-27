# Настройка AI-агентов (бесплатно)

## Получение Groq API ключа

1. **Регистрация:**
   - Перейдите на https://console.groq.com
   - Нажмите "Sign Up" 
   - Войдите через Google/GitHub

2. **Создание API ключа:**
   - Откройте https://console.groq.com/keys
   - Нажмите "Create API Key"
   - Скопируйте ключ (начинается с gsk_...)

3. **Настройка проекта:**
   ```bash
   # Откройте файл .env
   nano .env
   
   # Вставьте ваш ключ:
   GROQ_API_KEY=gsk_ваш_ключ_здесь
   
   # Сохраните (Ctrl+O, Enter, Ctrl+X)


