try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:  # Enables local smoke tests when dependencies cannot be installed offline.
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False
from app.agents.state import AgentState
from app.config import Settings
from app.services.ollama_client import OllamaClient
from app.tools.weather import OpenMeteoWeather, WeatherError, summarize_weather
import logging

logger = logging.getLogger(__name__)


WEATHER_WORDS = {
    "weather", "rain", "raining", "forecast", "temperature", "humidity", "wind", "weather tomorrow",
    "వాతావరణం", "వర్షం", "మಳೆ", "ಹವಾಮಾನ", "மழை", "வானிலை", "मौसम", "बारिश", "मौसम", "পाऊस",
}
CROP_WORDS = {"crop", "paddy", "rice", "wheat", "disease", "leaves", "పంట", "వ్యాధి", "फसल", "रोग", "பயிர்", "ಬೆಳೆ"}
MARKET_WORDS = {"price", "market", "mandi", "rate", "మార్కెట్ ధర", "ధర", "मंडी", "भाव", "விலை", "ಬೆಲೆ"}


LANGUAGE_NAMES = {"en": "English", "te": "Telugu", "hi": "Hindi", "bn": "Bengali", "kn": "Kannada", "ta": "Tamil", "mr": "Marathi", "gu": "Gujarati", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "ur": "Urdu"}
GENERAL_FALLBACKS = {
    "te": "నమస్కారం! నేను కిసాన్ దోస్త్‌ను. మీరు వాతావరణం, పంటలు లేదా మండీ ధరల గురించి అడగవచ్చు. వాతావరణ సూచన కోసం మీ గ్రామం లేదా జిల్లాను చెప్పండి, లేదా లొకేషన్ అనుమతిని ఇవ్వండి.",
    "hi": "नमस्ते! मैं किसान दोस्त हूँ। आप मुझसे मौसम, फसलों या मंडी भाव के बारे में पूछ सकते हैं। मौसम की जानकारी के लिए अपने गाँव या जिले का नाम बताएं, या लोकेशन की अनुमति दें।",
    "bn": "নমস্কার! আমি কিষাণ দোস্ত। আপনি আবহাওয়া, ফসল বা মাণ্ডির দাম সম্পর্কে জিজ্ঞাসা করতে পারেন। আবহাওয়ার পূর্বাভাসের জন্য আপনার গ্রাম বা জেলার নাম বলুন, অথবা লোকেশন অনুমতি দিন।",
    "kn": "ನಮಸ್ಕಾರ! ನಾನು ಕಿಸಾನ್ ದೋಸ್ತ್. ನೀವು ಹವಾಮಾನ, ಬೆಳೆಗಳು ಅಥವಾ ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳ ಬಗ್ಗೆ ಕೇಳಬಹುದು. ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಗಾಗಿ ನಿಮ್ಮ ಗ್ರಾಮ ಅಥವಾ ಜಿಲ್ಲೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ ಅಥವಾ ಸ್ಥಳ ಅನುಮತಿ ನೀಡಿ.",
    "ta": "வணக்கம்! நான் கிசான் தோஸ்த். வானிலை, பயிர்கள் அல்லது சந்தை விலைகள் பற்றி கேட்கலாம். வானிலை முன்னறிவிப்புக்கு உங்கள் கிராமம் அல்லது மாவட்டத்தின் பெயரைச் சொல்லுங்கள் அல்லது இருப்பிட அனுமதி வழங்குங்கள்.",
}
CROP_FALLBACKS = {
    "te": "పంటల గురించి సహాయం చేయగలను. దయచేసి పంట పేరు, పెరుగుదల దశ, కనిపిస్తున్న లక్షణాలను చెప్పండి. చికిత్స నిర్ణయాలకు స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.",
    "hi": "मैं फसलों के बारे में मदद कर सकता हूँ। कृपया फसल का नाम, बढ़ने की अवस्था और दिखाई देने वाले लक्षण बताएं। उपचार के लिए स्थानीय कृषि अधिकारी से सलाह लें।",
    "bn": "আমি ফসল সম্পর্কে সাহায্য করতে পারি। ফসলের নাম, বৃদ্ধির পর্যায় এবং লক্ষণগুলি জানান। চিকিৎসার সিদ্ধান্তের জন্য স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।",
}
MARKET_FALLBACKS = {
    "te": "ప్రస్తుతం ధృవీకరించిన మండీ ధరల సమాచారం అందుబాటులో లేదు. ధరను ఊహించి చెప్పను. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి.",
    "hi": "अभी सत्यापित मंडी भाव उपलब्ध नहीं हैं। मैं अनुमान से कीमत नहीं बताऊँगा। कृपया बाद में फिर कोशिश करें।",
    "bn": "এখন যাচাই করা মাণ্ডির দাম পাওয়া যাচ্ছে না। আমি অনুমান করে দাম বলব না। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
}

def _language_code(state: AgentState) -> str:
    code = str(state.get("language") or "en").lower().split("-")[0].split("_")[0]
    return code if code in LANGUAGE_NAMES else "en"

def _localized_weather_fallback(summary: str, language: str) -> str:
    """Return a concise, farmer-friendly weather answer while preserving measured values."""
    import re

    pattern = (
        r"Weather for (.+?) \(source: Open-Meteo\)\. Current temperature: ([\d.]+) °C\. "
        r"Relative humidity: ([\d.]+)%\. Current precipitation: ([\d.]+) mm\. "
        r"Wind speed: ([\d.]+) km/h\. Maximum precipitation probability today: ([\d.]+)%\. "
        r"Maximum precipitation probability tomorrow: ([\d.]+)%\. Forecast dates: (.+?)\. Forecasts can change\."
    )
    m = re.search(pattern, summary, re.IGNORECASE)
    if not m:
        # If the weather service changes its summary format, avoid guessing or corrupting data.
        return summary

    place, temp, humidity, rain, wind, today, tomorrow, _dates = m.groups()
    templates = {
        "en": (
            f"For {place}, rain is possible today, with a {today}% chance. "
            f"It is {temp}°C now; recorded precipitation is {rain} mm. "
            f"Tomorrow's rain chance is {tomorrow}%. Check local conditions before spraying."
        ),
        "te": (
            f"{place}లో ఈరోజు వర్షం పడే అవకాశం {today}% ఉంది. "
            f"ప్రస్తుతం ఉష్ణోగ్రత {temp}°C; నమోదైన వర్షపాతం {rain} మి.మీ. "
            f"రేపు వర్షం పడే అవకాశం {tomorrow}%. మందు పిచికారీ చేసే ముందు తాజా పరిస్థితులు చూసుకోండి."
        ),
        "hi": (
            f"{place} में आज बारिश की संभावना {today}% है। "
            f"अभी तापमान {temp}°C है और दर्ज वर्षा {rain} मिमी है। "
            f"कल बारिश की संभावना {tomorrow}% है। छिड़काव से पहले ताज़ा स्थिति जाँचें।"
        ),
        "bn": (
            f"{place}-এ আজ বৃষ্টির সম্ভাবনা {today}%. "
            f"এখন তাপমাত্রা {temp}°C এবং নথিভুক্ত বৃষ্টিপাত {rain} মিমি। "
            f"আগামীকাল বৃষ্টির সম্ভাবনা {tomorrow}%. স্প্রে করার আগে সর্বশেষ পরিস্থিতি দেখুন।"
        ),
        "kn": (
            f"{place}ದಲ್ಲಿ ಇಂದು ಮಳೆಯ ಸಾಧ್ಯತೆ {today}%. "
            f"ಈಗ ತಾಪಮಾನ {temp}°C; ದಾಖಲಾಗಿರುವ ಮಳೆ {rain} ಮಿಮೀ. "
            f"ನಾಳೆ ಮಳೆಯ ಸಾಧ್ಯತೆ {tomorrow}%. ಸಿಂಪಡಿಸುವ ಮೊದಲು ಇತ್ತೀಚಿನ ಸ್ಥಿತಿಯನ್ನು ಪರಿಶೀಲಿಸಿ."
        ),
        "ta": (
            f"{place}-இல் இன்று மழை பெய்யும் வாய்ப்பு {today}%. "
            f"தற்போதைய வெப்பநிலை {temp}°C; பதிவான மழைப்பொழிவு {rain} மிமீ. "
            f"நாளை மழை வாய்ப்பு {tomorrow}%. தெளிப்பதற்கு முன் தற்போதைய நிலையைச் சரிபார்க்கவும்."
        ),
    }
    return templates.get(language, summary)


def classify_intent(message: str) -> str:
    text = message.casefold()
    if any(word in text for word in WEATHER_WORDS):
        return "weather"
    if any(word in text for word in MARKET_WORDS):
        return "market"
    if any(word in text for word in CROP_WORDS):
        return "crop"
    return "general"


class KisanDostGraph:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.weather_tool = OpenMeteoWeather(settings)
        self.ollama = OllamaClient(settings)
        self.compiled = None
        if LANGGRAPH_AVAILABLE:
            graph = StateGraph(AgentState)
            graph.add_node("route", self._route)
            graph.add_node("weather", self._weather)
            graph.add_node("crop", self._crop)
            graph.add_node("market", self._market)
            graph.add_node("general", self._general)
            graph.add_node("answer", self._answer)
            graph.add_edge(START, "route")
            graph.add_conditional_edges("route", lambda state: state["intent"], {
                "weather": "weather", "crop": "crop", "market": "market", "general": "general"
            })
            for node in ("weather", "crop", "market", "general"):
                graph.add_edge(node, "answer")
            graph.add_edge("answer", END)
            self.compiled = graph.compile()

    async def _route(self, state: AgentState) -> dict:
        return {"intent": classify_intent(state["message"])}

    async def _weather(self, state: AgentState) -> dict:
        try:
            data = await self.weather_tool.for_request(
                state.get("location_name"), state.get("latitude"), state.get("longitude")
            )
            language = _language_code(state)
            summary = summarize_weather(data, language)
            summary = _localized_weather_fallback(summary, language)
            return {"weather": data, "response_text": summary, "model_used": "Open-Meteo + localized deterministic formatter"}
        except WeatherError as exc:
            language = _language_code(state)
            messages = {
                "te": "క్షమించండి, ప్రస్తుతం వాతావరణ సమాచారం పొందలేకపోయాను. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
                "hi": "क्षमा करें, अभी मौसम की जानकारी प्राप्त नहीं हो सकी। कृपया थोड़ी देर बाद फिर कोशिश करें।",
                "bn": "দুঃখিত, এই মুহূর্তে আবহাওয়ার তথ্য পাওয়া যায়নি। অনুগ্রহ করে কিছুক্ষণ পরে আবার চেষ্টা করুন।",
                "kn": "ಕ್ಷಮಿಸಿ, ಈಗ ಹವಾಮಾನ ಮಾಹಿತಿ ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
                "ta": "மன்னிக்கவும், இப்போது வானிலைத் தகவலைப் பெற முடியவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
            }
            logger.warning("Weather lookup failed: %s", exc)
            return {"weather": None, "response_text": messages.get(language, "Sorry, weather information is temporarily unavailable. Please try again shortly."), "model_used": "localized deterministic fallback"}

    async def _crop(self, state: AgentState) -> dict:
        language = _language_code(state)
        fallback = CROP_FALLBACKS.get(language, "I can help with crop questions, but the verified crop knowledge database is not connected yet. Please share the crop name, growth stage, and symptoms. For treatment decisions, confirm advice with a local agricultural extension officer.")
        return {"response_text": fallback, "model_used": "safe localized placeholder"}

    async def _market(self, state: AgentState) -> dict:
        language = _language_code(state)
        fallback = MARKET_FALLBACKS.get(language, "Live mandi prices are not connected yet, so I will not guess a price. Please try again after a verified market-price source is configured.")
        return {"response_text": fallback, "model_used": "safe localized placeholder"}

    async def _general(self, state: AgentState) -> dict:
        language = _language_code(state)
        fallback = GENERAL_FALLBACKS.get(language, "Namaste! I am Kisan Dost. You can ask me about weather, crops, or mandi prices. For a weather forecast, please share your village or district, or allow location access.")
        return {"response_text": fallback, "model_used": "localized deterministic fallback"}

    async def _answer(self, state: AgentState) -> dict:
        # Handle explicit language requests deterministically; don't echo the request back.
        message_lower = str(state.get("message", "")).casefold()
        if _language_code(state) == "te" and any(phrase in message_lower for phrase in ("తెలుగులో కావాలి", "తెలుగులో చెప్పు", "తెలుగులో సమాధానం", "తెలుగులో మాట్లాడండి")):
            return {"response_text": "తప్పకుండా, ఇకపై మీకు తెలుగులోనే సమాధానం ఇస్తాను. మీకు ఏ విషయం గురించి తెలుసుకోవాలి?", "model_used": "localized deterministic response"}
        # Weather is formatted deterministically so model language drift cannot change verified facts.
        if state.get("intent") == "weather":
            return {"response_text": _localized_weather_fallback(str(state.get("response_text", "")), _language_code(state)), "model_used": "Open-Meteo + localized deterministic formatter"}
        if not self.settings.ollama_enabled:
            if state.get("intent") == "weather":
                return {"response_text": _localized_weather_fallback(str(state.get("response_text", "")), _language_code(state)), "model_used": "localized deterministic fallback (Ollama disabled)"}
            return {"model_used": "localized deterministic fallback (Ollama disabled)"}
        language = _language_code(state)
        language_name = LANGUAGE_NAMES[language]
        history = state.get("conversation_history") or []
        history_text = "\n".join(
            f"{'Farmer' if item.get('role') == 'user' else 'Kisan Dost'}: {item.get('content', '')}"
            for item in history[-20:]
        )
        if not history_text:
            history_text = "No previous conversation."

        prompt = (
            f"Required output language: {language_name} (code {language}).\n"
            f"Previous conversation (oldest first):\n{history_text}\n"
            f"Use previous conversation to understand follow-up questions.\n"
            f"Farmer's original question: {state['message']}\n"
            f"Intent: {state.get('intent', 'general')}\n"
            f"Verified weather data, if any: {state.get('weather')}\n"
            f"Existing safe answer/context: {state.get('response_text', '')}\n"
            "Answer only in the required language. Do not echo or merely transliterate the farmer's question. "
            "Be conversational and farmer-friendly. Answer the actual question first, then summarize only the most useful facts. "
            "Usually use 2-4 short sentences and keep voice answers under about 30 seconds. "
            "Do not mention API/provider names, internal tools, raw data, or technical details unless the farmer asks. "
            "Avoid listing every measurement, forecast date, or repeating generic warnings. "
            "Do not invent weather, market prices, or crop-treatment facts. If verified data is missing, say so briefly. "
            "Give practical farming advice only when relevant, and distinguish forecasts from current observations. "
            "Return only the final answer text."
        )
        generated = await self.ollama.generate(
            prompt,
            f"You are Kisan Dost, a friendly, concise, multilingual assistant for Indian farmers. Always reply only in {language_name}. "
            "Answer the farmer's question first in simple everyday language. Summarize rather than repeat raw source details. "
            "Prefer 2-4 short sentences, suitable for speaking aloud. Do not mention API names or unnecessary technical details. "
            "Never invent current prices, forecasts, or crop-treatment facts. Never use English unless English is requested."
        )
        if generated and generated.strip():
            answer = generated.strip()
            # Reject obvious wrong-script outputs instead of returning English for a local-language request.
            expected_script = {
                "te": r"[\u0C00-\u0C7F]", "hi": r"[\u0900-\u097F]",
                "bn": r"[\u0980-\u09FF]", "kn": r"[\u0C80-\u0CFF]",
                "ta": r"[\u0B80-\u0BFF]", "ml": r"[\u0D00-\u0D7F]",
                "gu": r"[\u0A80-\u0AFF]", "pa": r"[\u0A00-\u0A7F]",
                "or": r"[\u0B00-\u0B7F]", "ur": r"[\u0600-\u06FF]",
            }
            import re
            pattern = expected_script.get(language)
            if pattern and not re.search(pattern, answer):
                logger.warning("Ollama returned text without expected %s script; using localized fallback", language)
                return {"model_used": "localized deterministic fallback (wrong-language model output)"}
            return {"response_text": answer, "model_used": self.settings.ollama_model}
        return {"model_used": "localized deterministic fallback (Ollama unavailable)"}

    async def invoke(self, state: AgentState) -> AgentState:
        if self.compiled is not None:
            return await self.compiled.ainvoke(state)
        # Minimal compatibility path for offline smoke tests only. Normal installs use LangGraph.
        result = dict(state)
        result.update(await self._route(result))
        handler = {
            "weather": self._weather,
            "crop": self._crop,
            "market": self._market,
            "general": self._general,
        }[result["intent"]]
        result.update(await handler(result))
        result.update(await self._answer(result))
        return result
