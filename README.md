# 🧮 GemMath AI

A web-based AI assistant that solves math problems, answers logical reasoning questions, and searches Wikipedia using **Google Gemma 2** via **Groq** and **LangChain**.

![GemMath Screenshot](https://placehold.co/600x300?text=GemMath+AI)

---

## 🚀 Features
- 🔢 **Solve Math Problems** with step-by-step explanations
- 📚 **Wikipedia Search** for general knowledge queries
- 🤖 **Logic Reasoning Tool** powered by LLM chain prompts
- 🧠 Uses **Google Gemma 2 model** through **Groq API**
- 🌐 Deployed using **Streamlit Cloud**

---

## 🔧 Installation & Local Run

```bash
git clone https://github.com/your-username/gemmath-ai.git
cd gemmath-ai
pip install -r requirements.txt
streamlit run app.py
```

> Make sure to paste your `Groq API Key` in the sidebar after launching.

---

## 🌍 Deployment (Free with Streamlit Cloud)
1. Push this project to a public GitHub repo
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo
4. Set the main file to `app.py`
5. ✅ Done!

---

## 📁 Project Structure
```
├── app.py
├── core
│   ├── agent_setup.py
│   └── prompt_templates.py
├── requirements.txt
└── .streamlit
    └── config.toml
```

---

## 📜 License
MIT License. Built for educational and demonstration purposes.

---

## ✨ Credits
- [Streamlit](https://streamlit.io)
- [Groq](https://groq.com)
- [LangChain](https://www.langchain.com)
- [Google Gemma 2](https://ai.google.dev/gemma)
