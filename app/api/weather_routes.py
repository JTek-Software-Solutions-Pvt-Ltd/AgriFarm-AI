from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.config import Settings
from app.postgres_db import get_farmer_details
from app.tools.weather import OpenMeteoWeather, WeatherError
from app.tools.agriculture_search import AgricultureSearch
from app.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "bn": "Bengali",
    "kn": "Kannada",
    "ta": "Tamil",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
}

SCRIPT_PATTERNS = {
    "te": r"[\u0C00-\u0C7F]",
    "hi": r"[\u0900-\u097F]",
    "bn": r"[\u0980-\u09FF]",
    "kn": r"[\u0C80-\u0CFF]",
    "ta": r"[\u0B80-\u0BFF]",
    "ml": r"[\u0D00-\u0D7F]",
    "gu": r"[\u0A80-\u0AFF]",
    "pa": r"[\u0A00-\u0A7F]",
    "or": r"[\u0B00-\u0B7F]",
    "ur": r"[\u0600-\u06FF]",
}

# Safe, localized fallback wording. Values are deliberately conditional and do
# not claim crop-specific diagnosis, fertilizer dose, pesticide treatment or
# irrigation quantity when the relevant field/soil data is absent.
FALLBACKS = {
    "en": {
        "crop_protection": "Use the forecast below to plan field work. If rain or strong wind is forecast, protect harvested produce and postpone spraying until conditions are suitable.",
        "irrigation_recommendations": "Check root-zone soil moisture before irrigating. Soil moisture and irrigation access were not supplied, so an irrigation quantity cannot be recommended.",
        "crop_care_recommendations": "Inspect the crop for visible stress and follow locally recommended practices for its growth stage. The growth stage and soil-test results were not supplied.",
        "preventive_measures": "Keep field drainage outlets clear where waterlogging is possible. Do not apply pesticides or fertilizer solely on the basis of this weather summary.",
        "summary": "Use the measured weather and forecast to plan field activities; confirm crop-stage and soil-specific decisions with the local KVK or agricultural extension officer.",
    },
    "te": {
        "crop_protection": "వాతావరణ సూచనను బట్టి పొలం పనులను ప్లాన్ చేయండి. వర్షం లేదా బలమైన గాలులు ఉంటే కోసిన పంటను కాపాడి, పరిస్థితులు అనుకూలంగా లేనప్పుడు పిచికారీ వాయిదా వేయండి.",
        "irrigation_recommendations": "నీరు పెట్టే ముందు వేర్ల ప్రాంతంలోని నేల తేమను పరిశీలించండి. నేల తేమ, నీటి సదుపాయం వివరాలు లేనందున నీటి పరిమాణాన్ని సూచించలేము.",
        "crop_care_recommendations": "పంటలో ఒత్తిడి లక్షణాలను గమనించి, పెరుగుదల దశకు స్థానికంగా సిఫారసు చేసిన పద్ధతులను అనుసరించండి. పంట దశ, నేల పరీక్ష వివరాలు అందలేదు.",
        "preventive_measures": "నీరు నిలిచే అవకాశం ఉన్న చోట పొలం కాలువలను శుభ్రంగా ఉంచండి. ఈ వాతావరణ వివరాల ఆధారంగానే పురుగుమందు లేదా ఎరువు వేయవద్దు.",
        "summary": "వాతావరణ సమాచారం ఆధారంగా పొలం పనులను ప్లాన్ చేయండి. పంట దశ, నేల ఆధారిత నిర్ణయాల కోసం స్థానిక కృషి విజ్ఞాన కేంద్రం లేదా వ్యవసాయ అధికారిని సంప్రదించండి.",
    },
    "hi": {
        "crop_protection": "मौसम पूर्वानुमान के अनुसार खेत के काम की योजना बनाएं। बारिश या तेज हवा की संभावना हो तो कटी उपज को सुरक्षित रखें और छिड़काव टालें।",
        "irrigation_recommendations": "सिंचाई से पहले जड़ क्षेत्र की मिट्टी की नमी जाँचें। मिट्टी की नमी और सिंचाई सुविधा की जानकारी नहीं है, इसलिए पानी की मात्रा बताना उचित नहीं है।",
        "crop_care_recommendations": "फसल में तनाव के लक्षण देखें और उसकी अवस्था के अनुसार स्थानीय कृषि सलाह अपनाएँ। फसल की अवस्था और मिट्टी जाँच के परिणाम उपलब्ध नहीं हैं।",
        "preventive_measures": "जहाँ पानी भरने की आशंका हो वहाँ खेत की नालियाँ साफ रखें। केवल इस मौसम जानकारी के आधार पर कीटनाशक या उर्वरक न डालें।",
        "summary": "मौसम की जानकारी के आधार पर खेत के काम तय करें और फसल तथा मिट्टी से जुड़े निर्णयों के लिए स्थानीय कृषि केंद्र से सलाह लें।",
    },
    "bn": {
        "crop_protection": "আবহাওয়ার পূর্বাভাস দেখে মাঠের কাজের পরিকল্পনা করুন। বৃষ্টি বা জোর হাওয়ার সম্ভাবনা থাকলে কাটা ফসল সুরক্ষিত রাখুন এবং স্প্রে স্থগিত রাখুন।",
        "irrigation_recommendations": "সেচের আগে শিকড়ের মাটির আর্দ্রতা পরীক্ষা করুন। মাটির আর্দ্রতা ও সেচের সুবিধার তথ্য নেই, তাই জলের পরিমাণ বলা যাচ্ছে না।",
        "crop_care_recommendations": "ফসলে চাপের লক্ষণ দেখুন এবং বৃদ্ধির পর্যায় অনুযায়ী স্থানীয় কৃষি পরামর্শ অনুসরণ করুন। বৃদ্ধির পর্যায় ও মাটি পরীক্ষার তথ্য নেই।",
        "preventive_measures": "জল জমার আশঙ্কা থাকলে নিকাশি নালা পরিষ্কার রাখুন। শুধু এই আবহাওয়ার তথ্যের ভিত্তিতে কীটনাশক বা সার প্রয়োগ করবেন না।",
        "summary": "আবহাওয়ার তথ্য দেখে মাঠের কাজ পরিকল্পনা করুন এবং ফসল ও মাটি-সংক্রান্ত সিদ্ধান্তে স্থানীয় কৃষি কেন্দ্রের পরামর্শ নিন।",
    },
    "kn": {
        "crop_protection": "ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಯ ಆಧಾರದ ಮೇಲೆ ಹೊಲದ ಕೆಲಸವನ್ನು ಯೋಜಿಸಿ. ಮಳೆ ಅಥವಾ ಬಲವಾದ ಗಾಳಿ ಇದ್ದರೆ ಕೊಯ್ಲಾದ ಉತ್ಪನ್ನವನ್ನು ರಕ್ಷಿಸಿ ಮತ್ತು ಸಿಂಪಡಣೆಯನ್ನು ಮುಂದೂಡಿ.",
        "irrigation_recommendations": "ನೀರಾವರಿಗೆ ಮೊದಲು ಬೇರು ಪ್ರದೇಶದ ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ. ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು ನೀರಾವರಿ ಸೌಲಭ್ಯದ ಮಾಹಿತಿ ಇಲ್ಲದ ಕಾರಣ ನೀರಿನ ಪ್ರಮಾಣ ಸೂಚಿಸಲಾಗುವುದಿಲ್ಲ.",
        "crop_care_recommendations": "ಬೆಳೆಯಲ್ಲಿ ಒತ್ತಡದ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ ಮತ್ತು ಬೆಳವಣಿಗೆಯ ಹಂತಕ್ಕೆ ಸ್ಥಳೀಯವಾಗಿ ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ ಅನುಸರಿಸಿ. ಬೆಳವಣಿಗೆಯ ಹಂತ ಮತ್ತು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.",
        "preventive_measures": "ನೀರು ನಿಲ್ಲುವ ಸಾಧ್ಯತೆ ಇರುವಲ್ಲಿ ಹೊಲದ ಕಾಲುವೆಗಳನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ. ಈ ಹವಾಮಾನ ಮಾಹಿತಿಯೊಂದರ ಆಧಾರದಲ್ಲಿ ಕೀಟನಾಶಕ ಅಥವಾ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.",
        "summary": "ಹವಾಮಾನ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಹೊಲದ ಕೆಲಸ ಯೋಜಿಸಿ; ಬೆಳೆ ಮತ್ತು ಮಣ್ಣಿನ ನಿರ್ಧಾರಗಳಿಗೆ ಸ್ಥಳೀಯ ಕೃಷಿ ಕೇಂದ್ರದ ಸಲಹೆ ಪಡೆಯಿರಿ.",
    },
    "mr": {
        "crop_protection": "हवामान अंदाजानुसार शेतातील कामांचे नियोजन करा. पाऊस किंवा जोरदार वारा असल्यास काढलेले पीक सुरक्षित ठेवा आणि फवारणी पुढे ढकला.",
        "irrigation_recommendations": "सिंचनापूर्वी मुळांच्या भागातील मातीतील ओलावा तपासा. मातीतील ओलावा आणि सिंचन सुविधेची माहिती नसल्याने पाण्याचे प्रमाण सांगता येत नाही.",
        "crop_care_recommendations": "पिकातील ताणाची लक्षणे तपासा आणि वाढीच्या अवस्थेनुसार स्थानिक कृषी सल्ला पाळा. वाढीची अवस्था आणि माती परीक्षणाची माहिती उपलब्ध नाही.",
        "preventive_measures": "पाणी साचण्याची शक्यता असलेल्या ठिकाणी निचरा मार्ग स्वच्छ ठेवा. फक्त या हवामान माहितीवरून कीटकनाशक किंवा खत वापरू नका.",
        "summary": "हवामानानुसार शेतीची कामे ठरवा आणि पीक व मातीविषयक निर्णयांसाठी स्थानिक कृषी केंद्राचा सल्ला घ्या.",
    },
    "gu": {
        "crop_protection": "હવામાન આગાહી પ્રમાણે ખેતરના કામનું આયોજન કરો. વરસાદ અથવા જોરદાર પવન હોય તો કાપેલી ઉપજ સુરક્ષિત રાખો અને છંટકાવ મુલતવી રાખો.",
        "irrigation_recommendations": "સિંચાઈ પહેલાં મૂળ વિસ્તારની જમીનની ભેજ તપાસો. જમીનની ભેજ અને સિંચાઈની સુવિધાની માહિતી ઉપલબ્ધ નથી, તેથી પાણીનું પ્રમાણ સૂચવી શકાય નહીં.",
        "crop_care_recommendations": "પાકમાં તાણનાં લક્ષણો તપાસો અને વૃદ્ધિની અવસ્થા મુજબ સ્થાનિક કૃષિ સલાહ અનુસરો. વૃદ્ધિની અવસ્થા અને જમીન પરીક્ષણની માહિતી ઉપલબ્ધ નથી.",
        "preventive_measures": "પાણી ભરાવાની શક્યતા હોય ત્યાં ખેતરના નિકાલના માર્ગો સાફ રાખો. માત્ર આ હવામાન માહિતીના આધારે જંતુનાશક અથવા ખાતર ન આપો.",
        "summary": "હવામાન પ્રમાણે ખેતરના કામનું આયોજન કરો અને પાક તથા જમીન સંબંધિત નિર્ણય માટે સ્થાનિક કૃષિ કેન્દ્રની સલાહ લો.",
    },
    "ml": {
        "crop_protection": "കാലാവസ്ഥാ പ്രവചനമനുസരിച്ച് കൃഷിപ്പണികൾ ആസൂത്രണം ചെയ്യുക. മഴയോ ശക്തമായ കാറ്റോ ഉണ്ടെങ്കിൽ വിളവെടുത്ത ഉൽപ്പന്നം സംരക്ഷിക്കുകയും തളിക്കൽ മാറ്റിവയ്ക്കുകയും ചെയ്യുക.",
        "irrigation_recommendations": "ജലസേചനത്തിന് മുമ്പ് വേരുകളുടെ ഭാഗത്തെ മണ്ണിലെ ഈർപ്പം പരിശോധിക്കുക. മണ്ണിലെ ഈർപ്പവും ജലസേചന സൗകര്യവും സംബന്ധിച്ച വിവരം ലഭ്യമല്ലാത്തതിനാൽ വെള്ളത്തിന്റെ അളവ് നിർദേശിക്കാനാവില്ല.",
        "crop_care_recommendations": "വിളയിലെ സമ്മർദ്ദ ലക്ഷണങ്ങൾ നിരീക്ഷിക്കുകയും വളർച്ചാ ഘട്ടത്തിന് അനുയോജ്യമായ പ്രാദേശിക കൃഷി നിർദേശങ്ങൾ പിന്തുടരുകയും ചെയ്യുക. വളർച്ചാ ഘട്ടവും മണ്ണ് പരിശോധനാ വിവരവും ലഭ്യമല്ല.",
        "preventive_measures": "വെള്ളക്കെട്ടിന് സാധ്യതയുള്ള സ്ഥലങ്ങളിൽ നീർചാലുകൾ വൃത്തിയായി സൂക്ഷിക്കുക. ഈ കാലാവസ്ഥാ വിവരത്തെ മാത്രം ആശ്രയിച്ച് കീടനാശിനിയോ വളമോ പ്രയോഗിക്കരുത്.",
        "summary": "കാലാവസ്ഥ അനുസരിച്ച് കൃഷിപ്പണികൾ ആസൂത്രണം ചെയ്യുക; വിളയെയും മണ്ണിനെയും കുറിച്ചുള്ള തീരുമാനങ്ങൾക്ക് പ്രാദേശിക കൃഷി കേന്ദ്രത്തിന്റെ ഉപദേശം തേടുക.",
    },
    "pa": {
        "crop_protection": "ਮੌਸਮ ਦੀ ਭਵਿੱਖਬਾਣੀ ਅਨੁਸਾਰ ਖੇਤ ਦੇ ਕੰਮ ਦੀ ਯੋਜਨਾ ਬਣਾਓ। ਮੀਂਹ ਜਾਂ ਤੇਜ਼ ਹਵਾ ਹੋਵੇ ਤਾਂ ਕੱਟੀ ਫਸਲ ਨੂੰ ਸੁਰੱਖਿਅਤ ਰੱਖੋ ਅਤੇ ਛਿੜਕਾਅ ਮੁਲਤਵੀ ਕਰੋ।",
        "irrigation_recommendations": "ਸਿੰਚਾਈ ਤੋਂ ਪਹਿਲਾਂ ਜੜ੍ਹਾਂ ਵਾਲੇ ਖੇਤਰ ਦੀ ਮਿੱਟੀ ਦੀ ਨਮੀ ਜਾਂਚੋ। ਮਿੱਟੀ ਦੀ ਨਮੀ ਅਤੇ ਸਿੰਚਾਈ ਸਹੂਲਤ ਦੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ, ਇਸ ਲਈ ਪਾਣੀ ਦੀ ਮਾਤਰਾ ਨਹੀਂ ਦੱਸੀ ਜਾ ਸਕਦੀ।",
        "crop_care_recommendations": "ਫਸਲ ਵਿੱਚ ਤਣਾਅ ਦੇ ਲੱਛਣ ਵੇਖੋ ਅਤੇ ਵਾਧੇ ਦੇ ਪੜਾਅ ਅਨੁਸਾਰ ਸਥਾਨਕ ਖੇਤੀ ਸਲਾਹ ਮੰਨੋ। ਵਾਧੇ ਦੇ ਪੜਾਅ ਅਤੇ ਮਿੱਟੀ ਜਾਂਚ ਦੀ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
        "preventive_measures": "ਜਿੱਥੇ ਪਾਣੀ ਖੜ੍ਹਾ ਹੋ ਸਕਦਾ ਹੈ ਉੱਥੇ ਨਿਕਾਸੀ ਨਾਲੀਆਂ ਸਾਫ਼ ਰੱਖੋ। ਸਿਰਫ਼ ਇਸ ਮੌਸਮੀ ਜਾਣਕਾਰੀ ਦੇ ਆਧਾਰ 'ਤੇ ਕੀਟਨਾਸ਼ਕ ਜਾਂ ਖਾਦ ਨਾ ਪਾਓ।",
        "summary": "ਮੌਸਮ ਅਨੁਸਾਰ ਖੇਤ ਦੇ ਕੰਮ ਤੈਅ ਕਰੋ ਅਤੇ ਫਸਲ ਤੇ ਮਿੱਟੀ ਬਾਰੇ ਫੈਸਲਿਆਂ ਲਈ ਸਥਾਨਕ ਖੇਤੀ ਕੇਂਦਰ ਦੀ ਸਲਾਹ ਲਵੋ।",
    },
    "or": {
        "crop_protection": "ପାଣିପାଗ ପୂର୍ବାନୁମାନ ଅନୁସାରେ କ୍ଷେତର କାମ ଯୋଜନା କରନ୍ତୁ। ବର୍ଷା କିମ୍ବା ଜୋର ପବନ ଥିଲେ କଟା ଫସଲକୁ ସୁରକ୍ଷିତ ରଖନ୍ତୁ ଏବଂ ସ୍ପ୍ରେ କାମ ସ୍ଥଗିତ ରଖନ୍ତୁ।",
        "irrigation_recommendations": "ଜଳସେଚନ ପୂର୍ବରୁ ମୂଳ ଅଞ୍ଚଳର ମାଟିର ଆର୍ଦ୍ରତା ଯାଞ୍ଚ କରନ୍ତୁ। ମାଟିର ଆର୍ଦ୍ରତା ଓ ଜଳସେଚନ ସୁବିଧାର ତଥ୍ୟ ନଥିବାରୁ ପାଣିର ପରିମାଣ କୁହାଯାଇପାରିବ ନାହିଁ।",
        "crop_care_recommendations": "ଫସଲରେ ଚାପର ଲକ୍ଷଣ ଦେଖନ୍ତୁ ଏବଂ ବୃଦ୍ଧି ଅବସ୍ଥା ଅନୁଯାୟୀ ସ୍ଥାନୀୟ କୃଷି ପରାମର୍ଶ ମାନନ୍ତୁ। ବୃଦ୍ଧି ଅବସ୍ଥା ଓ ମାଟି ପରୀକ୍ଷା ତଥ୍ୟ ମିଳିନାହିଁ।",
        "preventive_measures": "ଜଳ ଜମିବାର ସମ୍ଭାବନା ଥିବା ସ୍ଥାନରେ ନିଷ୍କାସନ ନାଳ ସଫା ରଖନ୍ତୁ। କେବଳ ଏହି ପାଣିପାଗ ତଥ୍ୟ ଆଧାରରେ କୀଟନାଶକ କିମ୍ବା ସାର ପ୍ରୟୋଗ କରନ୍ତୁ ନାହିଁ।",
        "summary": "ପାଣିପାଗ ଅନୁସାରେ କ୍ଷେତର କାମ ଯୋଜନା କରନ୍ତୁ ଏବଂ ଫସଲ ଓ ମାଟି ସମ୍ବନ୍ଧୀୟ ନିଷ୍ପତ୍ତି ପାଇଁ ସ୍ଥାନୀୟ କୃଷି କେନ୍ଦ୍ରର ପରାମର୍ଶ ନିଅନ୍ତୁ।",
    },
    "ur": {
        "crop_protection": "موسم کی پیش گوئی کے مطابق کھیت کے کام کی منصوبہ بندی کریں۔ بارش یا تیز ہوا کی صورت میں کٹی ہوئی پیداوار محفوظ رکھیں اور اسپرے مؤخر کریں۔",
        "irrigation_recommendations": "آبپاشی سے پہلے جڑوں کے علاقے کی مٹی میں نمی دیکھیں۔ مٹی کی نمی اور آبپاشی کی سہولت کی معلومات موجود نہیں، اس لیے پانی کی مقدار بتانا مناسب نہیں۔",
        "crop_care_recommendations": "فصل میں دباؤ کی علامات دیکھیں اور بڑھوتری کے مرحلے کے مطابق مقامی زرعی مشورے پر عمل کریں۔ بڑھوتری کے مرحلے اور مٹی کے ٹیسٹ کی معلومات دستیاب نہیں۔",
        "preventive_measures": "جہاں پانی جمع ہونے کا امکان ہو وہاں نکاسی کی نالیاں صاف رکھیں۔ صرف اس موسمی معلومات کی بنیاد پر کیڑے مار دوا یا کھاد استعمال نہ کریں۔",
        "summary": "موسم کے مطابق کھیت کے کام طے کریں اور فصل و مٹی سے متعلق فیصلوں کے لیے مقامی زرعی مرکز سے مشورہ لیں۔",
    },
    "ta": {
        "crop_protection": "வானிலை முன்னறிவிப்பைப் பார்த்து வயல் பணிகளைத் திட்டமிடுங்கள். மழை அல்லது பலத்த காற்று இருந்தால் அறுவடை செய்த விளைபொருளைப் பாதுகாத்து தெளிப்பதைத் தள்ளிவையுங்கள்.",
        "irrigation_recommendations": "நீர்ப்பாசனத்திற்கு முன் வேர் பகுதியில் மண்ணின் ஈரப்பதத்தைச் சரிபார்க்கவும். மண் ஈரப்பதம் மற்றும் நீர்ப்பாசன வசதி விவரம் இல்லாததால் நீரின் அளவை பரிந்துரைக்க முடியாது.",
        "crop_care_recommendations": "பயிரில் அழுத்த அறிகுறிகளைக் கண்காணித்து, வளர்ச்சி நிலைக்கு உள்ளூர் பரிந்துரைகளைப் பின்பற்றவும். வளர்ச்சி நிலை மற்றும் மண் பரிசோதனை விவரங்கள் இல்லை.",
        "preventive_measures": "நீர் தேங்கும் வாய்ப்புள்ள இடங்களில் வடிகால் கால்வாய்களைத் திறந்துவையுங்கள். இந்த வானிலைத் தகவலை மட்டும் வைத்து பூச்சிக்கொல்லி அல்லது உரம் இட வேண்டாம்.",
        "summary": "வானிலைத் தகவலின் அடிப்படையில் வயல் பணிகளைத் திட்டமிட்டு, பயிர் மற்றும் மண் தொடர்பான முடிவுகளுக்கு உள்ளூர் வேளாண்மை மையத்தின் ஆலோசனையைப் பெறுங்கள்.",
    },
}


class WeatherAdvisoryRequest(BaseModel):
    user_id: int = Field(gt=0)
    location_name: str | None = Field(default=None, min_length=2, max_length=200)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    crop_stage: str | None = Field(default=None, max_length=80)
    language: str = Field(default="en", max_length=12)
    forecast_days: int = Field(default=7, ge=1, le=16)

    @model_validator(mode="after")
    def validate_coordinates(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must be supplied together.")
        return self


MANDATORY_LIST_FIELDS = (
    "crop_protection",
    "irrigation_recommendations",
    "crop_care_recommendations",
    "preventive_measures",
)


def _language_code(value: str | None) -> str:
    code = (value or "en").lower().replace("_", "-").split("-")[0]
    return code if code in LANGUAGE_NAMES else "en"


NO_CROP_FALLBACKS = {
    "en": {
        "crop_protection": "No crop is currently recorded. Use the forecast to plan land preparation and other outdoor work; secure tools and materials before strong wind or rain.",
        "irrigation_recommendations": "No crop is selected, so an irrigation schedule or water quantity is not applicable. Check water availability and local conditions before preparing the field.",
        "crop_care_recommendations": "No planted crop is recorded. Review field access, drainage and water availability before the next planting decision; do not assume a crop or growth stage.",
        "preventive_measures": "Keep access paths and drainage outlets clear where rainfall is expected. Follow local weather warnings and avoid exposed outdoor work during thunder or lightning.",
        "summary": "This is a weather and field-preparation advisory because no crop is selected. Crop-specific recommendations will be considered after a crop and relevant field details are provided.",
    },
    "te": {
        "crop_protection": "ప్రస్తుతం ఏ పంటా నమోదు కాలేదు. వాతావరణ సూచనను బట్టి భూమి సిద్ధం చేయడం వంటి పనులను ప్లాన్ చేయండి; బలమైన గాలి లేదా వర్షానికి ముందు పనిముట్లు, సామగ్రిని భద్రపరచండి.",
        "irrigation_recommendations": "పంట ఎంపిక కాలేదు కాబట్టి నీటిపారుదల షెడ్యూల్ లేదా నీటి పరిమాణం వర్తించదు. పొలం సిద్ధం చేసే ముందు నీటి లభ్యత, స్థానిక పరిస్థితులను పరిశీలించండి.",
        "crop_care_recommendations": "నాటిన పంట నమోదు కాలేదు. తదుపరి సాగు నిర్ణయానికి ముందు పొలానికి వెళ్లే మార్గం, నీటి పారుదల, నీటి లభ్యతను పరిశీలించండి; పంట లేదా పెరుగుదల దశను ఊహించవద్దు.",
        "preventive_measures": "వర్షం ఉంటే పొలానికి వెళ్లే మార్గాలు, నీటి కాలువలను శుభ్రంగా ఉంచండి. స్థానిక వాతావరణ హెచ్చరికలను పాటించండి; ఉరుములు లేదా మెరుపుల సమయంలో బయట పని చేయవద్దు.",
        "summary": "పంట ఎంపిక కాలేదు కాబట్టి ఇది వాతావరణం మరియు పొలం సిద్ధతకు సంబంధించిన సూచన. పంట, అవసరమైన పొలం వివరాలు అందించిన తర్వాత పంట-ప్రత్యేక సూచనలు ఇవ్వగలం.",
    },
    "hi": {
        "crop_protection": "अभी कोई फसल दर्ज नहीं है। मौसम के अनुसार खेत की तैयारी और बाहरी काम तय करें; तेज हवा या बारिश से पहले औजार और सामग्री सुरक्षित रखें।",
        "irrigation_recommendations": "फसल नहीं चुनी गई है, इसलिए सिंचाई कार्यक्रम या पानी की मात्रा लागू नहीं होती। खेत तैयार करने से पहले पानी की उपलब्धता और स्थानीय स्थिति जाँचें।",
        "crop_care_recommendations": "कोई बोई हुई फसल दर्ज नहीं है। अगली बुवाई का निर्णय लेने से पहले खेत तक पहुँच, जल निकासी और पानी की उपलब्धता जाँचें; फसल या अवस्था का अनुमान न लगाएँ।",
        "preventive_measures": "बारिश की संभावना हो तो रास्ते और जल निकासी साफ रखें। स्थानीय मौसम चेतावनियों का पालन करें और गरज-चमक के दौरान खुले में काम न करें।",
        "summary": "फसल नहीं चुनी गई है, इसलिए यह मौसम और खेत की तैयारी संबंधी सलाह है। फसल और खेत की आवश्यक जानकारी मिलने पर फसल-विशिष्ट सलाह दी जा सकती है।",
    },
    "bn": {
        "crop_protection": "এখন কোনো ফসল নথিভুক্ত নেই। আবহাওয়া দেখে জমি প্রস্তুতি ও বাইরের কাজের পরিকল্পনা করুন; ঝোড়ো হাওয়া বা বৃষ্টির আগে সরঞ্জাম নিরাপদে রাখুন।",
        "irrigation_recommendations": "কোনো ফসল নির্বাচন করা হয়নি, তাই সেচের সময়সূচি বা জলের পরিমাণ প্রযোজ্য নয়। জমি প্রস্তুতির আগে জলের প্রাপ্যতা ও স্থানীয় পরিস্থিতি দেখুন।",
        "crop_care_recommendations": "কোনো রোপিত ফসল নথিভুক্ত নেই। পরবর্তী রোপণের সিদ্ধান্তের আগে জমিতে যাওয়ার পথ, নিকাশি ও জলের প্রাপ্যতা পরীক্ষা করুন; ফসল বা বৃদ্ধির পর্যায় অনুমান করবেন না।",
        "preventive_measures": "বৃষ্টির সম্ভাবনা থাকলে চলাচলের পথ ও নিকাশি পরিষ্কার রাখুন। স্থানীয় আবহাওয়ার সতর্কতা মানুন এবং বজ্রপাতের সময় খোলা জায়গায় কাজ করবেন না।",
        "summary": "কোনো ফসল নির্বাচন করা হয়নি, তাই এটি আবহাওয়া ও জমি প্রস্তুতির পরামর্শ। ফসল ও প্রয়োজনীয় জমির তথ্য দিলে ফসল-নির্দিষ্ট পরামর্শ দেওয়া যাবে।",
    },
    "kn": {
        "crop_protection": "ಪ್ರಸ್ತುತ ಯಾವುದೇ ಬೆಳೆ ದಾಖಲಾಗಿಲ್ಲ. ಹವಾಮಾನಕ್ಕೆ ಅನುಗುಣವಾಗಿ ಭೂಮಿ ಸಿದ್ಧತೆ ಮತ್ತು ಹೊರಾಂಗಣ ಕೆಲಸಗಳನ್ನು ಯೋಜಿಸಿ; ಬಲವಾದ ಗಾಳಿ ಅಥವಾ ಮಳೆಯ ಮೊದಲು ಉಪಕರಣಗಳನ್ನು ಸುರಕ್ಷಿತವಾಗಿಡಿ.",
        "irrigation_recommendations": "ಯಾವುದೇ ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡಿಲ್ಲ; ಆದ್ದರಿಂದ ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿ ಅಥವಾ ನೀರಿನ ಪ್ರಮಾಣ ಅನ್ವಯಿಸುವುದಿಲ್ಲ. ಹೊಲ ಸಿದ್ಧಪಡಿಸುವ ಮೊದಲು ನೀರಿನ ಲಭ್ಯತೆ ಮತ್ತು ಸ್ಥಳೀಯ ಪರಿಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ.",
        "crop_care_recommendations": "ನೆಟ್ಟ ಬೆಳೆ ದಾಖಲಾಗಿಲ್ಲ. ಮುಂದಿನ ಬಿತ್ತನೆ ನಿರ್ಧಾರಕ್ಕೂ ಮೊದಲು ಹೊಲದ ಪ್ರವೇಶ, ನೀರು ಹರಿವು ಮತ್ತು ನೀರಿನ ಲಭ್ಯತೆ ಪರಿಶೀಲಿಸಿ; ಬೆಳೆ ಅಥವಾ ಬೆಳವಣಿಗೆಯ ಹಂತವನ್ನು ಊಹಿಸಬೇಡಿ.",
        "preventive_measures": "ಮಳೆಯ ಸಾಧ್ಯತೆ ಇದ್ದರೆ ದಾರಿಗಳು ಮತ್ತು ನೀರು ಹರಿಯುವ ಕಾಲುವೆಗಳನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ. ಸ್ಥಳೀಯ ಹವಾಮಾನ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಪಾಲಿಸಿ; ಗುಡುಗು ಅಥವಾ ಮಿಂಚಿನ ವೇಳೆ ಹೊರಗೆ ಕೆಲಸ ಮಾಡಬೇಡಿ.",
        "summary": "ಯಾವುದೇ ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡಿಲ್ಲದ ಕಾರಣ ಇದು ಹವಾಮಾನ ಮತ್ತು ಹೊಲ ಸಿದ್ಧತೆ ಸಲಹೆಯಾಗಿದೆ. ಬೆಳೆ ಮತ್ತು ಅಗತ್ಯ ಹೊಲದ ವಿವರಗಳನ್ನು ನೀಡಿದ ಬಳಿಕ ಬೆಳೆ-ನಿರ್ದಿಷ್ಟ ಸಲಹೆ ನೀಡಬಹುದು.",
    },
    "ta": {
        "crop_protection": "தற்போது எந்தப் பயிரும் பதிவு செய்யப்படவில்லை. வானிலைக்கு ஏற்ப நிலத் தயாரிப்பு மற்றும் வெளிப்புறப் பணிகளைத் திட்டமிடுங்கள்; பலத்த காற்று அல்லது மழைக்கு முன் கருவிகளைப் பாதுகாப்பாக வையுங்கள்.",
        "irrigation_recommendations": "பயிர் தேர்ந்தெடுக்கப்படவில்லை; எனவே நீர்ப்பாசன அட்டவணை அல்லது நீரின் அளவு பொருந்தாது. நிலத்தைத் தயாரிக்கும் முன் நீர் கிடைப்பையும் உள்ளூர் நிலைமையையும் சரிபார்க்கவும்.",
        "crop_care_recommendations": "பயிரிடப்பட்ட பயிர் பதிவு செய்யப்படவில்லை. அடுத்த நடவு முடிவுக்கு முன் வயலுக்கான அணுகல், வடிகால் மற்றும் நீர் கிடைப்பைச் சரிபார்க்கவும்; பயிரையோ வளர்ச்சி நிலையையோ ஊகிக்க வேண்டாம்.",
        "preventive_measures": "மழை எதிர்பார்க்கப்பட்டால் பாதைகள் மற்றும் வடிகால் கால்வாய்களைச் சுத்தமாக வைத்திருங்கள். உள்ளூர் வானிலை எச்சரிக்கைகளைப் பின்பற்றுங்கள்; இடி அல்லது மின்னலின்போது திறந்தவெளியில் வேலை செய்ய வேண்டாம்.",
        "summary": "பயிர் தேர்ந்தெடுக்கப்படாததால் இது வானிலை மற்றும் நிலத் தயாரிப்பு ஆலோசனை. பயிர் மற்றும் தேவையான வயல் விவரங்கள் கிடைத்த பிறகு பயிர் சார்ந்த ஆலோசனை வழங்கலாம்.",
    },
    "mr": {
        "crop_protection": "सध्या कोणतेही पीक नोंदलेले नाही. हवामानानुसार जमीन तयार करण्याची आणि बाहेरील कामांची योजना करा; जोरदार वारा किंवा पावसापूर्वी साधने सुरक्षित ठेवा.",
        "irrigation_recommendations": "पीक निवडलेले नसल्याने सिंचनाचे वेळापत्रक किंवा पाण्याचे प्रमाण लागू होत नाही. शेत तयार करण्यापूर्वी पाण्याची उपलब्धता आणि स्थानिक परिस्थिती तपासा.",
        "crop_care_recommendations": "लागवड केलेले पीक नोंदलेले नाही. पुढील पेरणीपूर्वी शेतातील प्रवेश, निचरा आणि पाण्याची उपलब्धता तपासा; पीक किंवा वाढीची अवस्था गृहीत धरू नका.",
        "preventive_measures": "पावसाची शक्यता असल्यास रस्ते आणि निचऱ्याचे मार्ग मोकळे ठेवा. स्थानिक हवामान सूचना पाळा आणि मेघगर्जना किंवा विजेच्या वेळी मोकळ्या जागेत काम करू नका.",
        "summary": "पीक निवडलेले नसल्याने हा हवामान आणि शेत तयारीचा सल्ला आहे. पीक व आवश्यक शेतमाहिती मिळाल्यावर पीक-विशिष्ट सल्ला देता येईल.",
    },
    "gu": {
        "crop_protection": "હાલમાં કોઈ પાક નોંધાયેલો નથી. હવામાન મુજબ જમીન તૈયાર કરવાની અને બહારની કામગીરીની યોજના બનાવો; ભારે પવન અથવા વરસાદ પહેલાં સાધનો સુરક્ષિત રાખો.",
        "irrigation_recommendations": "કોઈ પાક પસંદ કરાયો નથી, તેથી સિંચાઈનું સમયપત્રક અથવા પાણીનું પ્રમાણ લાગુ પડતું નથી. ખેતર તૈયાર કરતાં પહેલાં પાણીની ઉપલબ્ધતા અને સ્થાનિક સ્થિતિ તપાસો.",
        "crop_care_recommendations": "વાવેલો પાક નોંધાયેલો નથી. આગામી વાવણીના નિર્ણય પહેલાં ખેતર સુધી પહોંચ, નિકાલ અને પાણીની ઉપલબ્ધતા તપાસો; પાક અથવા વૃદ્ધિ અવસ્થા માની ન લો.",
        "preventive_measures": "વરસાદની શક્યતા હોય ત્યારે રસ્તા અને નિકાલના માર્ગો ખુલ્લા રાખો. સ્થાનિક હવામાન ચેતવણીઓ અનુસરો અને ગાજવીજ દરમિયાન ખુલ્લામાં કામ ન કરો.",
        "summary": "પાક પસંદ કરાયો નથી, તેથી આ હવામાન અને ખેતર તૈયારી અંગેની સલાહ છે. પાક અને જરૂરી ખેતરની વિગતો મળ્યા પછી પાક-વિશિષ્ટ સલાહ આપી શકાશે.",
    },
    "ml": {
        "crop_protection": "നിലവിൽ വിളയൊന്നും രേഖപ്പെടുത്തിയിട്ടില്ല. കാലാവസ്ഥ അനുസരിച്ച് നിലം ഒരുക്കലും പുറംജോലികളും ആസൂത്രണം ചെയ്യുക; ശക്തമായ കാറ്റിനോ മഴയ്ക്കോ മുമ്പ് ഉപകരണങ്ങൾ സുരക്ഷിതമാക്കുക.",
        "irrigation_recommendations": "വിള തിരഞ്ഞെടുക്കാത്തതിനാൽ ജലസേചന സമയക്രമമോ വെള്ളത്തിന്റെ അളവോ ബാധകമല്ല. നിലം ഒരുക്കുന്നതിന് മുമ്പ് ജലലഭ്യതയും പ്രാദേശിക സാഹചര്യവും പരിശോധിക്കുക.",
        "crop_care_recommendations": "നട്ട വിള രേഖപ്പെടുത്തിയിട്ടില്ല. അടുത്ത നടീൽ തീരുമാനത്തിന് മുമ്പ് വയലിലേക്കുള്ള പ്രവേശനം, നീർചാൽ, ജലലഭ്യത എന്നിവ പരിശോധിക്കുക; വിളയോ വളർച്ചാഘട്ടമോ അനുമാനിക്കരുത്.",
        "preventive_measures": "മഴയ്ക്ക് സാധ്യതയുണ്ടെങ്കിൽ വഴികളും നീർചാലുകളും വൃത്തിയായി സൂക്ഷിക്കുക. പ്രാദേശിക കാലാവസ്ഥാ മുന്നറിയിപ്പുകൾ പാലിക്കുക; ഇടിമിന്നലുള്ളപ്പോൾ തുറസ്സായ സ്ഥലത്ത് ജോലി ചെയ്യരുത്.",
        "summary": "വിള തിരഞ്ഞെടുക്കാത്തതിനാൽ ഇത് കാലാവസ്ഥയും നിലം ഒരുക്കലും സംബന്ധിച്ച ഉപദേശമാണ്. വിളയും ആവശ്യമായ വയൽ വിവരങ്ങളും ലഭിച്ചാൽ വിള-നിർദ്ദിഷ്ട ഉപദേശം നൽകാം.",
    },
    "pa": {
        "crop_protection": "ਇਸ ਵੇਲੇ ਕੋਈ ਫਸਲ ਦਰਜ ਨਹੀਂ ਹੈ। ਮੌਸਮ ਅਨੁਸਾਰ ਜ਼ਮੀਨ ਦੀ ਤਿਆਰੀ ਅਤੇ ਬਾਹਰੀ ਕੰਮਾਂ ਦੀ ਯੋਜਨਾ ਬਣਾਓ; ਤੇਜ਼ ਹਵਾ ਜਾਂ ਮੀਂਹ ਤੋਂ ਪਹਿਲਾਂ ਸੰਦ ਸੁਰੱਖਿਅਤ ਰੱਖੋ।",
        "irrigation_recommendations": "ਕੋਈ ਫਸਲ ਨਹੀਂ ਚੁਣੀ ਗਈ, ਇਸ ਲਈ ਸਿੰਚਾਈ ਦਾ ਸਮਾਂ ਜਾਂ ਪਾਣੀ ਦੀ ਮਾਤਰਾ ਲਾਗੂ ਨਹੀਂ ਹੁੰਦੀ। ਖੇਤ ਤਿਆਰ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਪਾਣੀ ਦੀ ਉਪਲਬਧਤਾ ਅਤੇ ਸਥਾਨਕ ਹਾਲਾਤ ਜਾਂਚੋ।",
        "crop_care_recommendations": "ਬੀਜੀ ਹੋਈ ਫਸਲ ਦਰਜ ਨਹੀਂ ਹੈ। ਅਗਲੀ ਬਿਜਾਈ ਤੋਂ ਪਹਿਲਾਂ ਖੇਤ ਤੱਕ ਪਹੁੰਚ, ਨਿਕਾਸੀ ਅਤੇ ਪਾਣੀ ਦੀ ਉਪਲਬਧਤਾ ਜਾਂਚੋ; ਫਸਲ ਜਾਂ ਵਾਧੇ ਦੀ ਅਵਸਥਾ ਨਾ ਮੰਨੋ।",
        "preventive_measures": "ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ ਹੋਵੇ ਤਾਂ ਰਸਤੇ ਅਤੇ ਨਿਕਾਸੀ ਨਾਲੀਆਂ ਸਾਫ਼ ਰੱਖੋ। ਸਥਾਨਕ ਮੌਸਮੀ ਚੇਤਾਵਨੀਆਂ ਮੰਨੋ ਅਤੇ ਗਰਜ-ਚਮਕ ਦੌਰਾਨ ਖੁੱਲ੍ਹੇ ਵਿੱਚ ਕੰਮ ਨਾ ਕਰੋ।",
        "summary": "ਕੋਈ ਫਸਲ ਨਹੀਂ ਚੁਣੀ ਗਈ, ਇਸ ਲਈ ਇਹ ਮੌਸਮ ਅਤੇ ਖੇਤ ਦੀ ਤਿਆਰੀ ਬਾਰੇ ਸਲਾਹ ਹੈ। ਫਸਲ ਅਤੇ ਲੋੜੀਂਦੇ ਖੇਤ ਦੇ ਵੇਰਵੇ ਮਿਲਣ 'ਤੇ ਫਸਲ-ਵਿਸ਼ੇਸ਼ ਸਲਾਹ ਦਿੱਤੀ ਜਾ ਸਕਦੀ ਹੈ।",
    },
    "or": {
        "crop_protection": "ବର୍ତ୍ତମାନ କୌଣସି ଫସଲ ରେକର୍ଡ ହୋଇନାହିଁ। ପାଣିପାଗ ଅନୁସାରେ ଜମି ପ୍ରସ୍ତୁତି ଓ ବାହାର କାମ ଯୋଜନା କରନ୍ତୁ; ପ୍ରବଳ ପବନ କିମ୍ବା ବର୍ଷା ପୂର୍ବରୁ ଉପକରଣ ସୁରକ୍ଷିତ ରଖନ୍ତୁ।",
        "irrigation_recommendations": "କୌଣସି ଫସଲ ବଛାଯାଇନାହିଁ, ତେଣୁ ଜଳସେଚନ ସମୟସୂଚୀ କିମ୍ବା ଜଳ ପରିମାଣ ପ୍ରଯୁଜ୍ୟ ନୁହେଁ। ଜମି ପ୍ରସ୍ତୁତି ପୂର୍ବରୁ ଜଳ ଉପଲବ୍ଧତା ଓ ସ୍ଥାନୀୟ ପରିସ୍ଥିତି ଯାଞ୍ଚ କରନ୍ତୁ।",
        "crop_care_recommendations": "ଲଗାଯାଇଥିବା ଫସଲ ରେକର୍ଡ ହୋଇନାହିଁ। ପରବର୍ତ୍ତୀ ବୁଣା ନିଷ୍ପତ୍ତି ପୂର୍ବରୁ ଜମିକୁ ଯିବା ରାସ୍ତା, ଜଳ ନିଷ୍କାସନ ଓ ଜଳ ଉପଲବ୍ଧତା ଯାଞ୍ଚ କରନ୍ତୁ; ଫସଲ କିମ୍ବା ବୃଦ୍ଧି ଅବସ୍ଥା ଅନୁମାନ କରନ୍ତୁ ନାହିଁ।",
        "preventive_measures": "ବର୍ଷା ସମ୍ଭାବନା ଥିଲେ ରାସ୍ତା ଓ ନିଷ୍କାସନ ନାଳ ସଫା ରଖନ୍ତୁ। ସ୍ଥାନୀୟ ପାଣିପାଗ ସତର୍କତା ମାନନ୍ତୁ ଏବଂ ବଜ୍ରପାତ ସମୟରେ ଖୋଲା ସ୍ଥାନରେ କାମ କରନ୍ତୁ ନାହିଁ।",
        "summary": "କୌଣସି ଫସଲ ବଛାଯାଇନାହିଁ, ତେଣୁ ଏହା ପାଣିପାଗ ଓ ଜମି ପ୍ରସ୍ତୁତି ପରାମର୍ଶ। ଫସଲ ଓ ଆବଶ୍ୟକ ଜମି ସୂଚନା ମିଳିଲେ ଫସଲ-ନିର୍ଦ୍ଦିଷ୍ଟ ପରାମର୍ଶ ଦିଆଯାଇପାରିବ।",
    },
    "ur": {
        "crop_protection": "فی الحال کوئی فصل درج نہیں ہے۔ موسم کے مطابق زمین کی تیاری اور بیرونی کاموں کی منصوبہ بندی کریں؛ تیز ہوا یا بارش سے پہلے اوزار محفوظ رکھیں۔",
        "irrigation_recommendations": "کوئی فصل منتخب نہیں کی گئی، اس لیے آبپاشی کا شیڈول یا پانی کی مقدار لاگو نہیں ہوتی۔ زمین تیار کرنے سے پہلے پانی کی دستیابی اور مقامی حالات دیکھیں۔",
        "crop_care_recommendations": "لگائی گئی فصل درج نہیں ہے۔ اگلی کاشت کے فیصلے سے پہلے کھیت تک رسائی، نکاسی اور پانی کی دستیابی دیکھیں؛ فصل یا بڑھوتری کا مرحلہ فرض نہ کریں۔",
        "preventive_measures": "بارش کے امکان پر راستے اور نکاسی کی نالیاں صاف رکھیں۔ مقامی موسمی انتباہات پر عمل کریں اور گرج چمک کے دوران کھلی جگہ پر کام نہ کریں۔",
        "summary": "کوئی فصل منتخب نہیں کی گئی، اس لیے یہ موسم اور زمین کی تیاری سے متعلق مشورہ ہے۔ فصل اور ضروری کھیت کی معلومات ملنے کے بعد فصل کے مطابق مشورہ دیا جا سکتا ہے۔",
    },
}


def _fallback(language: str, has_crop: bool = True) -> dict[str, Any]:
    selected = FALLBACKS.get(language, FALLBACKS["en"])
    if not has_crop:
        selected = NO_CROP_FALLBACKS.get(language, NO_CROP_FALLBACKS["en"])
    return {
        "crop_protection": [selected["crop_protection"]],
        "irrigation_recommendations": [selected["irrigation_recommendations"]],
        "crop_care_recommendations": [selected["crop_care_recommendations"]],
        "preventive_measures": [selected["preventive_measures"]],
        "summary": selected["summary"],
    }


def _extract_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _has_expected_script(text: str, language: str) -> bool:
    pattern = SCRIPT_PATTERNS.get(language)
    if not pattern:
        return True
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not letters:
        return False
    matching = len(re.findall(pattern, text))
    return matching / max(len(letters), 1) >= 0.35


def _clean_strings(value: Any, fallback: list[str], language: str) -> list[str]:
    if not isinstance(value, list):
        return fallback.copy()
    items = [
        str(item).strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ][:5]
    if not items or any(not _has_expected_script(item, language) for item in items):
        return fallback.copy()
    return items


def _validate_advisory(value: dict[str, Any] | None, language: str) -> dict[str, Any]:
    fallback = _fallback(language)
    if not value:
        return fallback
    result = {
        key: _clean_strings(value.get(key), fallback[key], language)
        for key in MANDATORY_LIST_FIELDS
    }
    summary = value.get("summary")
    if (
        not isinstance(summary, str)
        or not summary.strip()
        or not _has_expected_script(summary, language)
    ):
        result["summary"] = fallback["summary"]
    else:
        result["summary"] = summary.strip()[:900]
    return result


def _normalise_location(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


STATE_ALIASES = {
    "andhrapradesh": {"andhrapradesh", "ap"},
    "telangana": {"telangana", "tg"},
}


def _district_variants(district: str | None) -> list[str]:
    value = str(district or "").strip()
    variants = [value] if value else []
    # The Nellore district is also stored under its former administrative name
    # in some farmer records. Use the common name only as a geocoding fallback.
    if "nellore" in value.casefold():
        variants.append("Nellore")
    return list(dict.fromkeys(variants))


def _location_queries(farmer: dict[str, Any], override: str | None) -> list[str]:
    if override:
        return [override]
    village = str(farmer.get("village") or "").strip()
    district_values = _district_variants(farmer.get("district"))
    state = str(farmer.get("state") or "").strip()
    country = str(farmer.get("country") or "India").strip()
    pincode = str(farmer.get("pincode") or "").strip()

    queries: list[str] = []
    for district in district_values:
        if village and district and state:
            queries.append(f"{village}, {district}, {state}, {country}")
        if village and pincode and state:
            queries.append(f"{village}, {pincode}, {state}, {country}")
        if district and state:
            queries.append(f"{district}, {state}, {country}")
    if village and state:
        queries.append(f"{village}, {state}, {country}")
    if pincode and state:
        queries.append(f"{pincode}, {state}, {country}")
    # Do not silently geocode to a state centroid. It is too coarse for farm advice.
    return list(dict.fromkeys(q for q in queries if q.strip()))


def _candidate_matches_farmer(place: dict[str, Any], farmer: dict[str, Any]) -> bool:
    country_code = str(place.get("country_code") or "").casefold()
    expected_country = str(farmer.get("country") or "India").casefold()
    if expected_country in {"india", "in"} and country_code and country_code != "in":
        return False

    expected_state = _normalise_location(farmer.get("state"))
    actual_state = _normalise_location(place.get("admin1"))
    if expected_state and actual_state:
        aliases = STATE_ALIASES.get(expected_state, {expected_state})
        if not any(actual_state == alias or actual_state in alias or alias in actual_state for alias in aliases):
            return False

    expected_district = _normalise_location(farmer.get("district"))
    actual_admin2 = _normalise_location(place.get("admin2"))
    if expected_district and actual_admin2:
        if "nellore" in expected_district and "nellore" in actual_admin2:
            return True
        # Do not reject a good village match because the provider uses a shortened
        # administrative name; state and country checks above still apply.
        if expected_district not in actual_admin2 and actual_admin2 not in expected_district:
            return False
    return True


async def _resolve_coordinates(
    service: OpenMeteoWeather,
    farmer: dict[str, Any],
    override: str | None = None,
) -> tuple[float, float, str]:
    for query in _location_queries(farmer, override):
        try:
            candidates = await service.geocode_candidates(query)
        except WeatherError as exc:
            logger.info("Geocoding query failed for saved farmer location: %s", exc)
            continue
        for place in candidates:
            if not _candidate_matches_farmer(place, farmer):
                continue
            label = ", ".join(
                str(x)
                for x in (
                    place.get("name"),
                    place.get("admin2"),
                    place.get("admin1"),
                    place.get("country"),
                )
                if x
            )
            return float(place["latitude"]), float(place["longitude"]), label or query
    raise WeatherError(
        "Could not resolve the saved village or district. Provide village, district and state in location_name, or verify the saved location."
    )


def _season_context(state: str | None, when: datetime) -> dict[str, str]:
    """Broad Indian season context; not a crop-specific sowing calendar."""
    month = when.month
    state_key = _normalise_location(state)
    # Broad southern/central Indian season windows. District and crop calendars
    # can differ; the response explicitly labels this as a broad context.
    if month in (6, 7, 8, 9):
        current, upcoming = "Kharif (monsoon)", "Rabi (post-monsoon/winter)"
    elif month in (10, 11, 12, 1, 2):
        current, upcoming = "Rabi (post-monsoon/winter)", "Zaid / summer"
    else:
        current, upcoming = "Zaid / summer", "Kharif (monsoon)"
    if state_key in {"andhrapradesh", "telangana"}:
        note = "Broad regional season context for Andhra Pradesh/Telangana; crop-specific sowing dates must be checked against the district KVK or state agricultural university calendar."
    else:
        note = "Broad Indian season context only; verify the district and crop calendar with the local KVK or state agricultural university."
    return {
        "current_season": current,
        "upcoming_season": upcoming,
        "season_basis": "broad_month_and_region_context",
        "season_note": note,
    }


def _as_float(items: Any, index: int) -> float | None:
    try:
        value = items[index]
        return float(value) if value is not None else None
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _temperature_summary(weather: dict[str, Any]) -> dict[str, Any]:
    current = weather.get("current", {})
    daily = weather.get("daily", {})
    hourly = weather.get("hourly", {})
    today_min = _as_float(daily.get("temperature_2m_min", []), 0)
    today_max = _as_float(daily.get("temperature_2m_max", []), 0)
    tomorrow_min = _as_float(daily.get("temperature_2m_min", []), 1)
    tomorrow_max = _as_float(daily.get("temperature_2m_max", []), 1)

    # Forecast range over the next 24 hours, based on hourly values returned by
    # Open-Meteo. The API's timestamps are local to its returned timezone.
    now_text = str(current.get("time") or "")
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    start_index = 0
    if now_text and times:
        for i, value in enumerate(times):
            if str(value) >= now_text:
                start_index = i
                break
    next_24 = [
        float(value)
        for value in temps[start_index:start_index + 24]
        if value is not None
    ]
    return {
        "current_c": current.get("temperature_2m"),
        "apparent_c": current.get("apparent_temperature"),
        "today_min_c": today_min,
        "today_max_c": today_max,
        "tomorrow_min_c": tomorrow_min,
        "tomorrow_max_c": tomorrow_max,
        "next_24h_min_c": min(next_24) if next_24 else None,
        "next_24h_max_c": max(next_24) if next_24 else None,
    }


def _hourly_alerts(weather: dict[str, Any]) -> list[dict[str, Any]]:
    """Create forecast watches from the next 1, 6 and 24 hourly records.

    Thresholds are operational watch thresholds, not official IMD warning
    criteria. Every item records the forecast interval and source type.
    """
    hourly = weather.get("hourly", {}) or {}
    times = hourly.get("time") or []
    current = weather.get("current", {}) or {}
    current_time = str(current.get("time") or "")
    fields = {
        "rain": hourly.get("precipitation") or [],
        "probability": hourly.get("precipitation_probability") or [],
        "temperature": hourly.get("temperature_2m") or [],
        "humidity": hourly.get("relative_humidity_2m") or [],
        "wind": hourly.get("wind_speed_10m") or [],
        "gust": hourly.get("wind_gusts_10m") or [],
        "code": hourly.get("weather_code") or [],
    }
    start = 0
    if current_time and times:
        start = next((i for i, value in enumerate(times) if str(value) >= current_time), 0)

    windows = ((1, "next_1_hour"), (6, "next_6_hours"), (24, "next_24_hours"))
    alerts: list[dict[str, Any]] = []
    for hours, label in windows:
        indexes = list(range(start, min(start + hours, len(times))))
        rows = [{key: _as_float(values, i) for key, values in fields.items()} for i in indexes]
        if not rows:
            continue

        def maximum(key: str, default: float = 0.0) -> float:
            values = [row[key] for row in rows if row.get(key) is not None]
            return max(values) if values else default

        def minimum(key: str) -> float | None:
            values = [row[key] for row in rows if row.get(key) is not None]
            return min(values) if values else None

        max_rain = maximum("rain")
        max_probability = maximum("probability")
        max_temp = maximum("temperature", default=float("-inf"))
        min_temp = minimum("temperature")
        max_gust = max(
            (row.get("gust") if row.get("gust") is not None else (row.get("wind") or 0))
            for row in rows
        )
        thunder = any(row.get("code") in (95, 96, 99) for row in rows)
        humid_hours = sum(1 for row in rows if (row.get("humidity") or 0) >= 85)
        first_time = str(times[indexes[0]]) if indexes else None
        last_time = str(times[indexes[-1]]) if indexes else None

        def add(kind: str, severity: str, description: str, action: str, impact: str, metrics: dict[str, Any]):
            alerts.append({
                "type": kind,
                "severity": severity,
                "window": label,
                "expected_time": first_time if first_time == last_time else f"{first_time} to {last_time}",
                "description": description,
                "potential_crop_impact": impact,
                "farmer_action": action,
                "observed_metrics": metrics,
                "source_type": "open_meteo_forecast_watch",
                "forecast_based": True,
                "official_warning": False,
            })

        if max_rain >= 7.5 or (max_rain >= 2 and max_probability >= 80):
            add("Heavy hourly rainfall watch", "HIGH" if max_rain >= 15 else "MEDIUM",
                f"Hourly forecast shows up to {max_rain:.1f} mm rainfall; maximum probability {max_probability:.0f}%.",
                "Clear field drainage, protect harvested produce and postpone spraying or top-dressing until rain has passed.",
                "Waterlogging, runoff and disruption to spraying, harvesting or drying may occur.",
                {"maximum_hourly_rainfall_mm": round(max_rain, 1), "maximum_rain_probability_percent": round(max_probability)})
        elif max_probability >= 70:
            add("Rainfall possibility", "LOW",
                f"Rain is possible in this forecast window; maximum probability is {max_probability:.0f}%.",
                "Check the latest hourly forecast before spraying, drying, harvesting or transporting produce.",
                "Rain may interrupt field operations.", {"maximum_rain_probability_percent": round(max_probability)})

        if thunder:
            add("Thunderstorm forecast watch", "HIGH",
                "The hourly forecast indicates thunderstorm conditions. This is a forecast watch, not an official lightning warning.",
                "Stop outdoor work during thunder. Move away from open fields, isolated trees, water, electric poles and metal equipment; use a substantial enclosed shelter.",
                "Lightning is a serious risk to farmers and livestock outdoors.", {"thunderstorm_weather_code": True})

        if max_temp != float("-inf") and max_temp >= 38:
            add("High temperature watch", "HIGH" if max_temp >= 42 else "MEDIUM",
                f"Hourly temperature may reach {max_temp:.1f}°C.",
                "Check for wilting and root-zone moisture; irrigate only when field conditions and crop guidance indicate a need. Avoid strenuous work during peak heat.",
                "Heat and moisture stress may increase, depending on crop stage and soil moisture.", {"maximum_temperature_c": round(max_temp, 1)})
        if min_temp is not None and min_temp <= 5:
            add("Low temperature watch", "MEDIUM",
                f"Hourly temperature may fall to {min_temp:.1f}°C.",
                "Monitor sensitive crop stages and seek local frost-protection advice.",
                "Cold stress may affect sensitive crops.", {"minimum_temperature_c": round(min_temp, 1)})
        if max_gust >= 35:
            add("Strong wind watch", "HIGH" if max_gust >= 50 else "MEDIUM",
                f"Wind gusts may reach {max_gust:.1f} km/h.",
                "Postpone pesticide spraying while gusts remain elevated; secure tarpaulins, shade nets and loose materials.",
                "Spray drift, lodging or damage to exposed structures may occur depending on crop and field conditions.", {"maximum_wind_gust_kmh": round(max_gust, 1)})
        if humid_hours >= min(4, hours):
            add("Sustained high humidity watch", "LOW",
                f"Relative humidity is forecast at or above 85% for {humid_hours} hour(s) in this window.",
                "Scout the crop for visible symptoms and improve field ventilation where practical. Humidity alone does not confirm a disease.",
                "Some fungal or bacterial disease risks can increase in susceptible crops; field inspection is required.", {"hours_relative_humidity_at_least_85": humid_hours})

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in alerts:
        key = (item["type"], item["window"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:30]



def _current_weather_alerts(weather: dict[str, Any]) -> list[dict[str, Any]]:
    """Raise operational watches for hazards already present in observations."""
    current = weather.get("current", {}) or {}
    observed_at = current.get("time")
    alerts: list[dict[str, Any]] = []

    def add(kind: str, severity: str, description: str, impact: str, action: str, metric: dict[str, Any]):
        alerts.append({
            "type": kind, "severity": severity, "window": "current_conditions",
            "expected_time": observed_at or "current observation",
            "description": description, "potential_crop_impact": impact,
            "farmer_action": action, "observed_metrics": metric,
            "source_type": "current_weather_observation",
            "forecast_based": False, "official_warning": False,
        })

    gust = current.get("wind_gusts_10m")
    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    code = current.get("weather_code")
    rain = current.get("precipitation")
    if isinstance(gust, (int, float)) and gust >= 35:
        add("Observed strong wind", "HIGH" if gust >= 50 else "MEDIUM",
            f"Latest observation reports wind gusts of {gust:.1f} km/h.",
            "Spray drift and damage to exposed covers or vulnerable crops may occur.",
            "Postpone pesticide spraying while gusts remain elevated and secure loose covers and materials.",
            {"wind_gust_kmh": round(gust, 1)})
    if isinstance(temp, (int, float)) and temp >= 38:
        add("Observed high temperature", "HIGH" if temp >= 42 else "MEDIUM",
            f"Latest observed temperature is {temp:.1f}°C.",
            "Heat and moisture stress may increase, depending on crop stage and soil moisture.",
            "Avoid strenuous outdoor work during peak heat; inspect the crop and check root-zone moisture.",
            {"temperature_c": round(temp, 1)})
    if code in (95, 96, 99):
        add("Observed thunderstorm condition", "HIGH",
            "The current weather code indicates thunderstorm conditions.",
            "Lightning can endanger farmers and livestock outdoors.",
            "Stop outdoor work and move to a substantial enclosed shelter; avoid trees, water, poles and wires.",
            {"weather_code": code})
    if isinstance(rain, (int, float)) and rain >= 7.5:
        add("Observed heavy precipitation", "MEDIUM" if rain < 15 else "HIGH",
            f"Latest observation reports {rain:.1f} mm precipitation.",
            "Field access and drainage may be affected.",
            "Check field drainage and protect harvested produce from getting wet.",
            {"precipitation_mm": round(rain, 1)})
    if isinstance(humidity, (int, float)) and humidity >= 90:
        add("Observed very high humidity", "LOW",
            f"Latest observation reports relative humidity of {humidity:.0f}%.",
            "Some crop disease risks may rise in susceptible crops, but humidity does not confirm disease.",
            "Scout for visible symptoms; do not apply treatment without diagnosis and local guidance.",
            {"relative_humidity_percent": round(humidity)})
    return alerts


def _daily_alerts(weather: dict[str, Any]) -> list[dict[str, Any]]:
    """Create 1–7 day forecast watches using daily forecast values."""
    daily = weather.get("daily", {}) or {}
    dates = daily.get("time") or []
    alerts: list[dict[str, Any]] = []
    for i, day in enumerate(dates):
        rain = _as_float(daily.get("precipitation_sum") or [], i)
        probability = _as_float(daily.get("precipitation_probability_max") or [], i)
        tmax = _as_float(daily.get("temperature_2m_max") or [], i)
        tmin = _as_float(daily.get("temperature_2m_min") or [], i)
        gust = _as_float(daily.get("wind_gusts_10m_max") or [], i)

        def add(kind: str, severity: str, description: str, impact: str, action: str, metrics: dict[str, Any]):
            alerts.append({
                "type": kind, "severity": severity, "window": str(day),
                "expected_time": str(day), "description": description,
                "potential_crop_impact": impact, "farmer_action": action,
                "observed_metrics": metrics,
                "source_type": "open_meteo_forecast_watch",
                "forecast_based": True, "official_warning": False,
            })

        if rain is not None and rain >= 20:
            add("Daily heavy rainfall watch", "HIGH" if rain >= 50 else "MEDIUM",
                f"Daily rainfall forecast is {rain:.1f} mm.",
                "Runoff, water stagnation and disruption to field operations are possible.",
                "Clear drainage outlets, protect harvested produce and avoid field operations on saturated soil.", {"rainfall_mm": round(rain, 1)})
        elif probability is not None and probability >= 60:
            add("Daily rainfall probability watch", "LOW" if probability < 80 else "MEDIUM",
                f"Maximum daily rain probability is {probability:.0f}% (forecast rainfall {rain if rain is not None else 'not available'} mm).",
                "Rain may interrupt spraying, drying, harvesting or transport.",
                "Check the latest hourly forecast before weather-sensitive field work.", {"rain_probability_percent": round(probability), "rainfall_mm": rain})
        if tmax is not None and tmax >= 38:
            add("Daily high temperature watch", "HIGH" if tmax >= 42 else "MEDIUM",
                f"Maximum temperature may reach {tmax:.1f}°C.",
                "Heat and moisture stress may increase, depending on crop stage and soil moisture.",
                "Inspect for wilting and check root-zone moisture; follow local crop-specific irrigation guidance.", {"maximum_temperature_c": round(tmax, 1)})
        if tmin is not None and tmin <= 5:
            add("Daily low temperature watch", "MEDIUM",
                f"Minimum temperature may fall to {tmin:.1f}°C.",
                "Cold stress may affect sensitive crops.",
                "Monitor sensitive crop stages and seek local frost-protection guidance.", {"minimum_temperature_c": round(tmin, 1)})
        if gust is not None and gust >= 35:
            add("Daily strong wind watch", "HIGH" if gust >= 50 else "MEDIUM",
                f"Maximum wind gust may reach {gust:.1f} km/h.",
                "Spray drift, lodging or physical damage may occur depending on crop and exposure.",
                "Avoid spraying during strong gusts and secure loose covers and materials.", {"maximum_wind_gust_kmh": round(gust, 1)})
    return alerts[:50]

def _merge_alerts(weather_alerts: list[dict[str, Any]], official_alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = []
    for item in official_alerts:
        combined.append({
            "type": item.get("type", "Official district weather warning"),
            "severity": str(item.get("severity", "MEDIUM")).upper(),
            "window": item.get("expected_time", "As stated by IMD"),
            "expected_time": item.get("expected_time", "As stated by IMD"),
            "description": item.get("message", "An official district warning is available."),
            "potential_crop_impact": "Follow the official warning for local impacts.",
            "farmer_action": item.get("precaution", "Follow IMD and local administration instructions."),
            "source_type": "official_district_warning",
            "forecast_based": False,
        })
    combined.extend(weather_alerts)
    return combined[:40]


def _weather_display(weather: dict[str, Any]) -> dict[str, Any]:
    current = weather.get("current", {})
    daily = weather.get("daily", {})
    hourly = weather.get("hourly", {})
    return {
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "rainfall_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_gust_kmh": current.get("wind_gusts_10m"),
            "weather_code": current.get("weather_code"),
            "observed_at": current.get("time"),
        },
        "temperature_forecast": _temperature_summary(weather),
        "daily_forecast": [
            {
                "date": daily.get("time", [])[i],
                "minimum_temperature_c": _as_float(daily.get("temperature_2m_min") or [], i),
                "maximum_temperature_c": _as_float(daily.get("temperature_2m_max") or [], i),
                "rainfall_mm": _as_float(daily.get("precipitation_sum") or [], i),
                "rain_probability_percent": _as_float(daily.get("precipitation_probability_max") or [], i),
                "maximum_wind_gust_kmh": _as_float(daily.get("wind_gusts_10m_max") or [], i),
            }
            for i in range(min(len(daily.get("time") or []), 7))
        ],
        "hourly_forecast_available": bool(hourly.get("time")),
        "timezone": weather.get("timezone"),
        "forecast_retrieved_at": weather.get("retrieved_at_utc"),
    }


def _weather_summary(weather: dict[str, Any], language: str) -> str:
    current = weather.get("current", {})
    temps = _temperature_summary(weather)
    values = []
    if temps["current_c"] is not None:
        values.append(f"Current temperature: {temps['current_c']}°C")
    if temps["today_min_c"] is not None and temps["today_max_c"] is not None:
        values.append(f"today's range: {temps['today_min_c']}–{temps['today_max_c']}°C")
    if current.get("relative_humidity_2m") is not None:
        values.append(f"humidity: {current['relative_humidity_2m']}%")
    if current.get("precipitation") is not None:
        values.append(f"current precipitation: {current['precipitation']} mm")
    if current.get("wind_speed_10m") is not None:
        values.append(f"wind: {current['wind_speed_10m']} km/h")
    return "; ".join(values) + "." if values else "Weather observations are temporarily unavailable."


async def _generate_advisory(
    ollama: OllamaClient,
    farmer: dict[str, Any],
    crop: dict[str, Any] | None,
    weather: dict[str, Any],
    season: dict[str, str],
    alerts: list[dict[str, Any]],
    research: dict[str, Any],
    language: str,
    crop_stage: str | None,
) -> tuple[dict[str, Any], str]:
    fallback = _fallback(language, has_crop=bool(crop))
    if crop is None:
        return fallback, "localized_no_crop_weather_and_field_preparation_fallback"
    if not ollama.settings.ollama_enabled:
        return fallback, "localized_safe_fallback_ollama_disabled"

    current = weather.get("current", {})
    daily = weather.get("daily", {})
    hourly = weather.get("hourly", {})
    # Research snippets are unverified leads. The model may use them to explain
    # what needs local verification, but not to assert crop suitability/treatment.
    context = {
        "farmer_scenario": "existing_crop" if crop else "no_crop",
        "location": {
            "state": farmer.get("state"),
            "district": farmer.get("district"),
            "village": farmer.get("village"),
        },
        "crop": crop,
        "crop_growth_stage": crop_stage,
        "soil_information": {
            "soil_type": farmer.get("soil_type"),
            "soil_ph": farmer.get("soil_ph"),
            "soil_moisture": farmer.get("soil_moisture"),
            "nitrogen": farmer.get("nitrogen"),
            "phosphorus": farmer.get("phosphorus"),
            "potassium": farmer.get("potassium"),
            "note": "Fields absent from the existing database query are unavailable; do not infer them.",
        },
        "season": season,
        "current_weather": current,
        "temperature_forecast": _temperature_summary(weather),
        "daily_forecast": daily,
        "next_24_hour_forecast": {
            key: (hourly.get(key) or [])[:24]
            for key in ("time", "temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation", "weather_code", "wind_speed_10m", "wind_gusts_10m")
        },
        "weather_alerts": [
            {k: v for k, v in alert.items() if k != "source_type"}
            for alert in alerts
        ],
        "agricultural_research_leads": [
            {
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "url": item.get("url"),
                "verification_status": "unverified_search_lead",
            }
            for item in (research.get("results") or [])[:6]
        ],
    }
    schema = {
        "crop_protection": ["string"],
        "irrigation_recommendations": ["string"],
        "crop_care_recommendations": ["string"],
        "preventive_measures": ["string"],
        "summary": "string",
    }
    system = (
        f"You are Kisan Dost, an Indian farmer advisory assistant. Respond only in {LANGUAGE_NAMES[language]} "
        f"(language code {language}). Return only valid JSON matching the supplied schema. "
        "Use only the supplied weather and farmer data. Make advice specific to the observed temperature, "
        "forecast, rainfall, humidity, wind, location, crop name and crop stage when known. "
        "Do not invent soil type, soil moisture, pH, NPK, crop stage, irrigation access, disease diagnosis, "
        "sowing dates, fertilizer doses, pesticide names or chemical doses. Search snippets are unverified leads, "
        "not evidence; do not use them to validate crop suitability or treatments. "
        "Weather codes and forecasts are not official severe-weather warnings. Distinguish a forecast watch "
        "from an official warning. Do not claim lightning is certain. "
        "For irrigation, refer to measured soil moisture if present; otherwise tell the farmer to check soil "
        "moisture before deciding and do not state a water quantity. For fertilizer, never prescribe a dose "
        "without soil-test and crop-stage evidence. For pests/diseases, describe scouting only; weather alone "
        "cannot diagnose disease. If the farmer has no crop, do not invent or recommend crop names. "
        "Use the current and upcoming season only as broad context, not a district crop calendar. "
        "Generate actionable recommendations that respond to this farmer's actual weather data. "
        "Do not repeat the same generic sentence in every field. Keep each list to 1-4 concise strings."
    )
    prompt = (
        "Create the farmer-facing advisory. Return only this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\nFARMER AND WEATHER CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, default=str)
    )
    try:
        raw = await ollama.generate(prompt, system)
    except Exception:
        logger.exception("Ollama advisory generation failed")
        return fallback, "localized_safe_fallback_ollama_error"

    generated = _extract_json(raw)
    validated = _validate_advisory(generated, language)
    if generated is None:
        return validated, "localized_safe_fallback_invalid_model_json"
    # If every model field fell back, identify the response as fallback for diagnostics.
    model_output_used = any(
        validated[key] != fallback[key] for key in MANDATORY_LIST_FIELDS
    ) or validated["summary"] != fallback["summary"]
    return validated, "ollama" if model_output_used else "localized_safe_fallback_validation"



def _localized_weather_summary(weather: dict[str, Any], language: str) -> str:
    """Build a concise weather summary using localized labels where available."""
    current = weather.get("current", {})
    temps = _temperature_summary(weather)
    labels = {
        "en": ("Current temperature", "Today's range", "Humidity", "Rainfall", "Wind", "Weather observations are temporarily unavailable."),
        "te": ("ప్రస్తుత ఉష్ణోగ్రత", "ఈ రోజు ఉష్ణోగ్రత పరిధి", "తేమ", "వర్షపాతం", "గాలి వేగం", "ప్రస్తుతం వాతావరణ సమాచారం అందుబాటులో లేదు."),
        "hi": ("वर्तमान तापमान", "आज का तापमान", "नमी", "वर्षा", "हवा", "मौसम की जानकारी अभी उपलब्ध नहीं है।"),
        "bn": ("বর্তমান তাপমাত্রা", "আজকের তাপমাত্রার সীমা", "আর্দ্রতা", "বৃষ্টিপাত", "বাতাস", "আবহাওয়ার তথ্য সাময়িকভাবে পাওয়া যাচ্ছে না।"),
        "kn": ("ಪ್ರಸ್ತುತ ತಾಪಮಾನ", "ಇಂದಿನ ತಾಪಮಾನದ ವ್ಯಾಪ್ತಿ", "ತೇವಾಂಶ", "ಮಳೆ", "ಗಾಳಿ", "ಹವಾಮಾನ ಮಾಹಿತಿ ತಾತ್ಕಾಲಿಕವಾಗಿ ಲಭ್ಯವಿಲ್ಲ."),
        "ta": ("தற்போதைய வெப்பநிலை", "இன்றைய வெப்பநிலை வரம்பு", "ஈரப்பதம்", "மழைப்பொழிவு", "காற்று", "வானிலைத் தகவல் தற்காலிகமாக கிடைக்கவில்லை."),
        "mr": ("सध्याचे तापमान", "आजची तापमान मर्यादा", "आर्द्रता", "पाऊस", "वारा", "हवामानाची माहिती तात्पुरती उपलब्ध नाही."),
        "gu": ("વર્તમાન તાપમાન", "આજની તાપમાન મર્યાદા", "ભેજ", "વરસાદ", "પવન", "હવામાનની માહિતી અત્યારે ઉપલબ્ધ નથી."),
        "ml": ("നിലവിലെ താപനില", "ഇന്നത്തെ താപനില പരിധി", "ഈർപ്പം", "മഴ", "കാറ്റ്", "കാലാവസ്ഥാ വിവരം താൽക്കാലികമായി ലഭ്യമല്ല."),
        "pa": ("ਮੌਜੂਦਾ ਤਾਪਮਾਨ", "ਅੱਜ ਦੇ ਤਾਪਮਾਨ ਦੀ ਹੱਦ", "ਨਮੀ", "ਮੀਂਹ", "ਹਵਾ", "ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ ਅਸਥਾਈ ਤੌਰ 'ਤੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।"),
        "or": ("ବର୍ତ୍ତମାନ ତାପମାତ୍ରା", "ଆଜିର ତାପମାତ୍ରା ସୀମା", "ଆର୍ଦ୍ରତା", "ବର୍ଷା", "ପବନ", "ପାଣିପାଗ ସୂଚନା ବର୍ତ୍ତମାନ ଉପଲବ୍ଧ ନାହିଁ।"),
        "ur": ("موجودہ درجہ حرارت", "آج کے درجہ حرارت کی حد", "نمی", "بارش", "ہوا", "موسم کی معلومات عارضی طور پر دستیاب نہیں ہیں۔"),
    }
    current_label, range_label, humidity_label, rain_label, wind_label, unavailable = labels.get(language, labels["en"])
    values: list[str] = []
    if temps["current_c"] is not None:
        values.append(f"{current_label}: {temps['current_c']}°C")
    if temps["today_min_c"] is not None and temps["today_max_c"] is not None:
        values.append(f"{range_label}: {temps['today_min_c']}–{temps['today_max_c']}°C")
    if current.get("relative_humidity_2m") is not None:
        values.append(f"{humidity_label}: {current['relative_humidity_2m']}%")
    if current.get("precipitation") is not None:
        values.append(f"{rain_label}: {current['precipitation']} mm")
    if current.get("wind_speed_10m") is not None:
        values.append(f"{wind_label}: {current['wind_speed_10m']} km/h")
    return "; ".join(values) + "." if values else unavailable


def _build_action_plan(
    alerts: list[dict[str, Any]],
    weather: dict[str, Any],
    language: str,
    crop: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build distinct, non-repeating actions for immediate, 24h and 3-7d.

    A single hazard may span several forecast windows. It is represented once
    per time band, and the future band uses a monitoring instruction rather
    than repeating the immediate action verbatim.
    """
    del language  # English-first implementation; localization is intentionally deferred.
    immediate: list[dict[str, Any]] = []
    next_24_hours: list[dict[str, Any]] = []
    next_3_7_days: list[dict[str, Any]] = []
    crop_name = str((crop or {}).get("name") or "the crop")

    def hazard_group(alert: dict[str, Any]) -> str:
        label = str(alert.get("type") or "").casefold()
        if "wind" in label or "gust" in label:
            return "wind"
        if "rain" in label or "precipitation" in label:
            return "rain"
        if "thunder" in label or "lightning" in label:
            return "thunderstorm"
        if "heat" in label or "temperature" in label or "cold" in label:
            return "temperature"
        if "humidity" in label:
            return "humidity"
        return label or "weather"

    def make_entry(alert: dict[str, Any], *, future_monitor: bool = False) -> dict[str, Any]:
        severity = str(alert.get("severity") or "LOW").upper()
        action = str(alert.get("farmer_action") or alert.get("description") or "Review the latest local forecast.")
        group = hazard_group(alert)
        if future_monitor:
            if group == "wind":
                action = "Recheck wind gusts before resuming spraying; proceed only when conditions are suitable and the product label permits it."
            elif group == "rain":
                action = ("Recheck rainfall timing before scheduling field preparation or moving equipment." if not crop else "Recheck the latest rainfall timing before irrigation, spraying, harvesting or drying produce.")
            elif group == "thunderstorm":
                action = "Monitor official local weather updates and keep people away from exposed fields during thunder or lightning."
            elif group == "temperature":
                action = ("Monitor the temperature forecast and schedule outdoor field work during safer, cooler periods." if not crop else "Monitor the temperature forecast and inspect the crop for heat or cold stress during the forecast period.")
            elif group == "humidity":
                action = ("Monitor local weather conditions; humidity alone does not establish a crop disease risk when no crop is selected." if not crop else "Continue field scouting for disease symptoms; humidity alone is not a diagnosis or reason to spray.")
            else:
                action = "Recheck the latest forecast before weather-sensitive field operations."
        return {
            "action": action,
            "reason": alert.get("description"),
            "priority": "high" if severity in {"HIGH", "SEVERE", "CRITICAL"} else "medium" if severity in {"MEDIUM", "MODERATE"} else "low",
            "alert_type": alert.get("type"),
            "expected_time": alert.get("expected_time"),
            "source_type": alert.get("source_type", "forecast_based"),
        }

    seen_immediate: set[str] = set()
    seen_24h: set[str] = set()
    seen_3_7d: set[str] = set()
    daily_dates = (weather.get("daily", {}) or {}).get("time") or []
    today = str(daily_dates[0]) if daily_dates else None
    tomorrow = str(daily_dates[1]) if len(daily_dates) > 1 else None

    for alert in alerts or []:
        group = hazard_group(alert)
        windows = alert.get("windows") or [{
            "window": alert.get("window"),
            "expected_time": alert.get("expected_time"),
            "source_type": alert.get("source_type"),
        }]
        immediate_windows = [w for w in windows if str(w.get("window") or "") in {"current_conditions", "next_1_hour", "next_6_hours"} or (today and str(w.get("window")) == today)]
        future_24_windows = [w for w in windows if str(w.get("window") or "") == "next_24_hours" or (tomorrow and str(w.get("window")) == tomorrow)]
        future_daily_windows = [w for w in windows if str(w.get("window") or "") not in {"current_conditions", "next_1_hour", "next_6_hours", "next_24_hours"} and str(w.get("window") or "") != (today or "") and str(w.get("window") or "") != (tomorrow or "")]

        if immediate_windows and group not in seen_immediate:
            selected = dict(alert)
            selected["expected_time"] = immediate_windows[0].get("expected_time") or alert.get("expected_time")
            selected["source_type"] = immediate_windows[0].get("source_type") or alert.get("source_type")
            immediate.append(make_entry(selected))
            seen_immediate.add(group)

        if future_24_windows and group not in seen_24h:
            selected = dict(alert)
            selected["expected_time"] = future_24_windows[0].get("expected_time") or alert.get("expected_time")
            selected["source_type"] = future_24_windows[0].get("source_type") or alert.get("source_type")
            next_24_hours.append(make_entry(selected, future_monitor=group in seen_immediate))
            seen_24h.add(group)

        # Date-specific daily forecast windows are handled below one day at a
        # time. Avoid adding a combined alert action for the same dates.
        daily_window_dates = {str(value) for value in daily_dates[2:]}
        non_daily_future_windows = [
            window for window in future_daily_windows
            if str(window.get("window") or "") not in daily_window_dates
        ]
        if non_daily_future_windows and group not in seen_3_7d:
            selected = dict(alert)
            selected["expected_time"] = non_daily_future_windows[0].get("expected_time") or alert.get("expected_time")
            selected["source_type"] = non_daily_future_windows[0].get("source_type") or alert.get("source_type")
            next_3_7_days.append(make_entry(selected, future_monitor=(group in seen_immediate or group in seen_24h)))
            seen_3_7d.add(group)

    # Defensive current-observation fallback if an alert generator did not emit
    # an observed-wind event for a gust that crosses the operational threshold.
    current = weather.get("current", {}) or {}
    gust_now = current.get("wind_gusts_10m")
    if isinstance(gust_now, (int, float)) and gust_now >= 35 and "wind" not in seen_immediate:
        immediate.append({
            "action": "Postpone pesticide spraying while current wind gusts remain elevated; secure loose tarpaulins and materials.",
            "reason": f"Current measured wind gust is {gust_now:.1f} km/h.",
            "priority": "medium" if gust_now < 50 else "high",
            "expected_time": current.get("time"),
            "source_type": "current_weather_observation",
        })
        seen_immediate.add("wind")

    daily = weather.get("daily", {}) or {}
    dates = daily.get("time") or []
    for index in range(1, min(len(dates), 8)):
        date = str(dates[index])
        # Tomorrow belongs to the 24-hour band. The remaining dates belong to 3-7d.
        if tomorrow and date == tomorrow:
            continue
        rain = _as_float(daily.get("precipitation_sum") or [], index)
        probability = _as_float(daily.get("precipitation_probability_max") or [], index)
        tmax = _as_float(daily.get("temperature_2m_max") or [], index)
        gust = _as_float(daily.get("wind_gusts_10m_max") or [], index)
        day_actions: list[str] = []
        reasons: list[str] = []
        if (rain is not None and rain >= 10) or (probability is not None and probability >= 60):
            day_actions.append("Check the latest hourly rain forecast before scheduling field preparation or moving equipment." if not crop else "Check the latest hourly rain forecast before irrigation, spraying, harvesting or drying produce.")
            reasons.append(f"rain {rain if rain is not None else 'n/a'} mm; probability {probability if probability is not None else 'n/a'}%")
        if tmax is not None and tmax >= 35:
            day_actions.append("Plan outdoor field work during cooler hours and ensure drinking water is available." if not crop else "Inspect the field for signs of heat stress and check root-zone soil moisture during the cooler part of the day.")
            reasons.append(f"maximum temperature {tmax:.1f}°C")
        if gust is not None and gust >= 35:
            day_actions.append("Secure loose equipment, temporary covers and materials before strong gusts." if not crop else "Avoid spraying during strong gusts and secure loose covers or harvested produce.")
            reasons.append(f"maximum gust {gust:.1f} km/h")
        if day_actions:
            next_3_7_days.append({
                "action": " ".join(dict.fromkeys(day_actions)),
                "reason": f"{date}: " + "; ".join(reasons),
                "priority": "medium" if (rain or 0) >= 10 or (probability or 0) >= 80 or (gust or 0) >= 50 or (tmax or 0) >= 40 else "low",
                "expected_time": date,
                "crop_context": crop_name if crop else None,
                "source_type": "daily_forecast",
            })

    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in items:
            action_key = re.sub(r"\W+", " ", str(entry.get("action") or "").casefold()).strip()
            # Identical instructions on different dates are separate forecast
            # events, so include expected_time in the de-duplication key.
            dedupe_key = f"{entry.get('expected_time') or 'unscheduled'}::{action_key}"
            if action_key and dedupe_key not in seen:
                seen.add(dedupe_key)
                output.append(entry)
        return output[:10]

    immediate = compact(immediate)
    next_24_hours = compact(next_24_hours)
    next_3_7_days = compact(next_3_7_days)

    if not immediate:
        immediate = [{
            "action": (
                f"No configured high-priority weather trigger is present in the next 6 hours. Inspect {crop_name} for visible stress and check local field conditions before spraying or irrigation."
                if crop else
                "No configured high-priority weather trigger is present in the next 6 hours. Check local field conditions before spraying, irrigation or other field operations."
            ),
            "reason": "Based on the currently available hourly forecast and observation data.",
            "priority": "low", "source_type": "weather_and_crop_observation",
        }]
    if not next_24_hours:
        next_24_hours = [{
            "action": "Review the next 24-hour forecast before weather-sensitive field work and update the plan if conditions change.",
            "reason": "No additional threshold-triggered event was found for the next 24 hours.",
            "priority": "low", "source_type": "forecast_review",
        }]
    if not next_3_7_days:
        next_3_7_days = [{
            "action": (
                f"The available 3–7 day forecast shows no configured weather trigger. Continue routine field scouting of {crop_name} and review the forecast daily."
                if crop else
                "The available 3–7 day forecast shows no configured weather trigger. Review the forecast daily and plan field operations accordingly."
            ),
            "reason": "No threshold-triggered rainfall, heat or wind event found in the available daily forecast.",
            "priority": "low", "source_type": "daily_forecast_review",
        }]
    return {"immediate": immediate, "next_24_hours": next_24_hours, "next_3_7_days": next_3_7_days}


CROP_IDENTITY_REFERENCE = {
    "cowpea": {
        "scientific_name": "Vigna unguiculata",
        "aliases": {
            "alasande gram", "alasandalu", "alasandulu", "bobbarlu", "bobbarelu",
            "alsando", "alsande", "cowpea", "cow gram", "cowgram", "lobia",
        },
        "sources": [
            {
                "title": "Telangana Directorate of Economics and Statistics – Agricultural Statistics at a Glance, crop glossary",
                "url": "https://des.telangana.gov.in/publications/Agricultural_at_glance_2015-16.pdf",
                "supports": "The glossary lists Vigna catjang / cowgram-cowpea with Telugu names Alasandalu or Bobbarlu.",
            },
            {
                "title": "ICAR-CCARI – Cowpea (Alsando)",
                "url": "https://ccari.res.in/dss/cowpea.html",
                "supports": "Identifies Alsando as cowpea and gives scientific name Vigna unguiculata.",
            },
            {
                "title": "ICAR-CCARI – Vegetable cowpea",
                "url": "https://ccari.res.in/dss/vegetable%20cowpea.html",
                "supports": "Confirms cowpea is also cultivated for tender vegetable pods, as well as dry pulse; therefore name alone does not prove production purpose.",
            },
        ],
    }
}


def _normalise_crop_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(value or "").casefold()).strip()


def _resolve_crop_identity(
    crop: dict[str, Any] | None,
    farmer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a saved local crop name to a botanical crop identity.

    The resolver deliberately separates botanical identity from end-use
    (grain/pulse versus fresh vegetable pods). A local alias can identify the
    species without proving how this particular farmer grows or sells it.
    """
    if not crop:
        return {
            "status": "no_crop_selected",
            "common_name": None,
            "scientific_name": None,
            "production_purpose": "unknown",
            "purpose_confidence": "not_assessed",
            "confidence": "none",
            "evidence": [],
        }

    name = str(crop.get("name") or "").strip()
    norm_name = _normalise_crop_name(name)
    category = _normalise_crop_name(crop.get("category"))
    details = _normalise_crop_name(crop.get("crop_details"))
    combined = f"{norm_name} {category} {details}".strip()

    for common_name, reference in CROP_IDENTITY_REFERENCE.items():
        aliases = {_normalise_crop_name(alias) for alias in reference["aliases"]}
        name_match = norm_name in aliases or any(
            len(alias) >= 6 and (norm_name.startswith(alias + " ") or alias in norm_name)
            for alias in aliases
        )
        if not name_match:
            continue

        purpose = "unknown"
        purpose_confidence = "not_assessed"
        if re.search(r"\b(grain|pulse|dry seed|dry bean|dal)\b", combined):
            purpose, purpose_confidence = "grain_or_pulse", "medium"
        elif re.search(r"\b(vegetable|fresh pod|tender pod|yard long|yardlong)\b", combined):
            purpose, purpose_confidence = "vegetable_pod", "medium"
        elif re.search(r"\b(fodder|dual purpose|dual-purpose)\b", combined):
            purpose, purpose_confidence = "dual_or_fodder", "medium"

        return {
            "status": "species_resolved_purpose_unresolved" if purpose == "unknown" else "species_and_purpose_resolved",
            "common_name": "Cowpea",
            "scientific_name": reference["scientific_name"],
            "matched_database_name": name,
            "production_purpose": purpose,
            "purpose_confidence": purpose_confidence,
            "confidence": "high_for_species_local_name_match",
            "resolution_basis": ["saved_crop_name_alias", "Telangana_crop_glossary", "ICAR_crop_reference"],
            "evidence": reference["sources"],
            "farmer_confirmation_required": purpose == "unknown",
            "note": (
                "The crop name identifies cowpea, but does not establish whether this farmer grows it for dry grain, fresh vegetable pods or another use. Do not select a purpose-specific package until corroborating farm records, sales/market records or crop imagery are available."
                if purpose == "unknown" else
                "Production purpose is inferred from explicit crop category/details; confirm against the farmer's actual crop and market records before applying purpose-specific inputs."
            ),
        }

    return {
        "status": "unresolved",
        "common_name": name or None,
        "scientific_name": crop.get("scientific_name"),
        "production_purpose": "unknown",
        "purpose_confidence": "not_assessed",
        "confidence": "low",
        "resolution_basis": ["saved_crop_record_only"],
        "evidence": [],
        "farmer_confirmation_required": True,
        "note": "The saved crop name could not be matched confidently to a verified crop alias. Keep crop-specific input recommendations disabled.",
    }


def _alert_group_key(alert: dict[str, Any]) -> str:
    label = str(alert.get("type") or "").casefold()
    if "wind" in label or "gust" in label:
        return "strong_wind"
    if "rain" in label or "precipitation" in label:
        return "rainfall"
    if "thunder" in label or "lightning" in label:
        return "thunderstorm"
    if "temperature" in label or "heat" in label or "cold" in label:
        return "temperature"
    if "humidity" in label:
        return "humidity"
    return label or "weather_event"


def _deduplicate_weather_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group repeated forecast windows without concatenating duplicate prose.

    One event keeps a single concise description, crop impact and farmer
    action. Every distinct time window and its source metadata is preserved.
    """
    grouped: dict[tuple[str, bool], dict[str, Any]] = {}
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "SEVERE": 4, "CRITICAL": 5}
    for alert in alerts:
        key = (_alert_group_key(alert), bool(alert.get("official_warning")))
        window_info = {
            "window": alert.get("window"),
            "expected_time": alert.get("expected_time"),
            "forecast_based": bool(alert.get("forecast_based")),
            "official_warning": bool(alert.get("official_warning")),
            "source_type": alert.get("source_type"),
        }
        item = grouped.get(key)
        if item is None:
            item = dict(alert)
            item["windows"] = [window_info]
            item["observed_metrics"] = dict(alert.get("observed_metrics") or {})
            grouped[key] = item
            continue

        windows = item.setdefault("windows", [])
        if window_info not in windows:
            windows.append(window_info)

        incoming_severity = str(alert.get("severity") or "LOW").upper()
        current_severity = str(item.get("severity") or "LOW").upper()
        if severity_rank.get(incoming_severity, 1) > severity_rank.get(current_severity, 1):
            item["severity"] = incoming_severity
            # If the stronger event has more specific text, use that single
            # description/action rather than appending duplicate sentences.
            for field in ("description", "farmer_action", "potential_crop_impact"):
                if alert.get(field):
                    item[field] = alert[field]

        metrics = item.setdefault("observed_metrics", {})
        for metric, value in (alert.get("observed_metrics") or {}).items():
            existing = metrics.get(metric)
            if existing is None:
                metrics[metric] = value
            elif isinstance(value, (int, float)) and isinstance(existing, (int, float)):
                metrics[metric] = max(existing, value)

        # The top-level flag means the alert is wholly forecast-derived.
        # A mixed observation + forecast alert must therefore remain False;
        # the exact provenance of every time window is retained below.
        item["forecast_based"] = bool(windows) and all(
            bool(w.get("forecast_based")) for w in windows
        )
        item["official_warning"] = any(
            bool(w.get("official_warning")) for w in windows
        )
        sources = list(dict.fromkeys(
            w.get("source_type") for w in windows if w.get("source_type")
        ))
        item["source_types"] = sources
        item["source_type"] = (
            sources[0] if len(sources) == 1 else
            "combined_observation_and_forecast" if sources else "weather_data"
        )
        times = list(dict.fromkeys(
            str(w.get("expected_time")) for w in windows if w.get("expected_time")
        ))
        # Keep expected_time a stable string for existing clients. Consumers
        # needing every time must use expected_times / windows.
        item["expected_times"] = times
        item["expected_time"] = times[0] if times else None
        item["window"] = "combined_windows" if len(windows) > 1 else item.get("window")

    return list(grouped.values())


def _crop_specific_monitoring(crop: dict[str, Any] | None, weather: dict[str, Any], identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return weather-linked scouting actions without pretending to diagnose.

    Agronomic recommendations requiring a verified local package of practices
    remain gated until the original authoritative document is available.
    """
    if not crop:
        return {
            "status": "no_existing_crop",
            "crop_name": None,
            "weather_based_actions": [],
            "verified_agronomic_recommendations": [],
            "evidence_status": "no_crop_selected",
        }

    name = str(crop.get("name") or "Unknown crop")
    identity = identity or _resolve_crop_identity(crop)
    current = weather.get("current", {}) or {}
    daily = weather.get("daily", {}) or {}
    actions: list[dict[str, Any]] = [{
        "action": f"Scout {name} fields and record visible wilting, yellowing, leaf spots, stem damage or unusual pest activity; a symptom report or field inspection is needed before diagnosis.",
        "trigger": "routine_crop_scouting",
        "priority": "low",
        "basis": "saved_crop_identity_and_farmer_field_observation",
    }]
    humidity = current.get("relative_humidity_2m")
    gust = current.get("wind_gusts_10m")
    rain_values = daily.get("precipitation_sum") or []
    rain_next_48h = sum(float(v) for v in rain_values[1:3] if isinstance(v, (int, float)))
    if isinstance(humidity, (int, float)) and humidity >= 80:
        actions.append({
            "action": f"Inspect {name} for visible leaf or stem symptoms after humid periods. High humidity is a risk indicator only and does not confirm disease.",
            "trigger": "high_current_relative_humidity",
            "observed_value": f"{humidity:.0f}%",
            "priority": "low",
            "basis": "current_weather_observation",
        })
    if isinstance(gust, (int, float)) and gust >= 35:
        actions.append({
            "action": f"Avoid pesticide spraying on {name} while wind gusts remain elevated; check the product label and local extension advice before any application.",
            "trigger": "elevated_wind_gust",
            "observed_value": f"{gust:.1f} km/h",
            "priority": "medium",
            "basis": "current_weather_observation",
        })
    if rain_next_48h >= 10:
        actions.append({
            "action": f"Check field drainage around {name} and postpone weather-sensitive operations if the latest forecast confirms rainfall.",
            "trigger": "forecast_rain_next_48h",
            "observed_value": f"{rain_next_48h:.1f} mm forecast across the next two daily periods",
            "priority": "medium",
            "basis": "daily_weather_forecast",
        })
    return {
        "status": "weather_based_monitoring_only",
        "crop_name": name,
        "crop_identity": identity,
        "weather_based_actions": actions,
        "verified_agronomic_recommendations": [],
        "evidence_status": "species_identity_resolved_but_local_package_and_production_purpose_not_verified",
        "note": "The botanical crop identity can be resolved from the saved local name, but grain versus vegetable production purpose and the district-specific package of practices are not yet established. Weather-linked monitoring is enabled; purpose-specific fertilizer, pesticide, irrigation quantity and sowing recommendations remain gated.",
    }

def build_weather_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/v1/weather", tags=["Weather Advisory"])
    weather_service = OpenMeteoWeather(settings)
    agriculture_search = AgricultureSearch()
    ollama = OllamaClient(settings)

    @router.post("/advisory")
    async def weather_advisory(body: WeatherAdvisoryRequest) -> dict[str, Any]:
        language = _language_code(body.language)
        try:
            record = await get_farmer_details(body.user_id)
        except Exception as exc:
            logger.exception("Could not retrieve farmer profile")
            raise HTTPException(
                status_code=503,
                detail="Farmer information is temporarily unavailable.",
            ) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="Farmer not found or inactive.")

        farmer = dict(record)
        crop_id = farmer.get("crop_id")
        crop_name = farmer.get("crop_name")
        crop = (
            {
                "id": crop_id,
                "name": crop_name,
                "scientific_name": farmer.get("scientific_name"),
                "category": farmer.get("category"),
                "database_season": farmer.get("season"),
                "crop_details": farmer.get("crop_details"),
            }
            if crop_id is not None and crop_name
            else None
        )
        crop_identity = _resolve_crop_identity(crop, farmer)
        if crop and crop_identity.get("scientific_name") and not crop.get("scientific_name"):
            crop["scientific_name"] = crop_identity["scientific_name"]

        try:
            if body.latitude is not None and body.longitude is not None:
                latitude, longitude = body.latitude, body.longitude
                location_label = body.location_name or "Farmer-selected location"
            elif farmer.get("latitude") is not None and farmer.get("longitude") is not None:
                latitude, longitude = float(farmer["latitude"]), float(farmer["longitude"])
                location_label = ", ".join(
                    str(x) for x in (
                        farmer.get("village"), farmer.get("district"),
                        farmer.get("state"), farmer.get("country") or "India",
                    ) if x
                )
            else:
                latitude, longitude, location_label = await _resolve_coordinates(
                    weather_service, farmer, body.location_name
                )
            weather = await weather_service.forecast(
                latitude, longitude, location_label, body.forecast_days
            )
        except WeatherError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Weather retrieval failed")
            raise HTTPException(
                status_code=502,
                detail="Weather data is temporarily unavailable.",
            ) from exc

        now = datetime.now(timezone.utc)
        season = _season_context(farmer.get("state"), now)
        current = weather.get("current", {})
        daily_alerts = _daily_alerts(weather)
        hourly_alerts = _hourly_alerts(weather)

        official_alerts: list[dict[str, Any]] = []
        district = str(farmer.get("district") or "")
        imd_method = getattr(weather_service, "imd_district_alerts", None)
        if district and callable(imd_method):
            try:
                official_alerts = await imd_method(district)
            except Exception:
                logger.warning("Official district warning lookup failed", exc_info=True)

        current_alerts = _current_weather_alerts(weather)
        # Collapse overlapping current/hourly/daily detections into one event
        # while retaining each forecast window in the `windows` array.
        deduplicated_alerts = _deduplicate_weather_alerts(current_alerts + hourly_alerts + daily_alerts)
        alerts = _merge_alerts(deduplicated_alerts, official_alerts)

        try:
            research = await agriculture_search.research(
                farmer=farmer,
                crop=crop,
                current_weather=current,
                daily_forecast=weather.get("daily", {}),
            )
        except Exception:
            logger.warning("Agricultural search failed; continuing without search leads", exc_info=True)
            research = {"results": [], "research_status": "unavailable"}

        advisory, advisory_source = await _generate_advisory(
            ollama=ollama,
            farmer=farmer,
            crop=crop,
            weather=weather,
            season=season,
            alerts=alerts,
            research=research,
            language=language,
            crop_stage=body.crop_stage,
        )

        weather_display = _weather_display(weather)
        advisory_payload: dict[str, Any] = {
            "crop_name": crop_name if crop else None,
            "crop_stage": body.crop_stage,
            "current_season": season["current_season"],
            "upcoming_season": season["upcoming_season"],
            "season_note": season["season_note"],
            "weather_summary": _localized_weather_summary(weather, language),
            "weather_alerts": [
                {k: v for k, v in item.items() if k not in {"source_type"}}
                for item in alerts
            ],
            **{key: advisory[key] for key in MANDATORY_LIST_FIELDS},
            "summary": advisory["summary"],
            "crop_identity": crop_identity,
            "crop_specific_recommendations": _crop_specific_monitoring(crop, weather, crop_identity),
        }
        if crop is None:
            # Do not recommend crops from search snippets. The current source
            # integration discovers leads but does not verify original documents.
            advisory_payload["suitable_crops"] = []
            advisory_payload["alternative_crops"] = []
            advisory_payload["crop_recommendation_status"] = "awaiting_verified_regional_agronomic_evidence"
            advisory_payload["crop_recommendation_note"] = (
                "No crop is recommended until an original authoritative source supports the crop, district, season, soil and water conditions. Search snippets alone are not sufficient evidence."
            )

        return {
            "success": True,
            "farmer_type": "existing_crop" if crop else "no_crop",
            "language": language,
            "last_updated": now.isoformat(),
            "next_update": (now + timedelta(minutes=30)).isoformat(),
            "location": {
                "village": farmer.get("village"),
                "district": farmer.get("district"),
                "state": farmer.get("state"),
                "label": location_label,
            },
            "weather": weather_display,
            "crop_information": {
                "crop_name": crop_name if crop else None,
                "scientific_name": crop.get("scientific_name") if crop else None,
                "crop_identity_status": crop_identity.get("status"),
                "production_purpose": crop_identity.get("production_purpose"),
                "database_crop_season": crop.get("database_season") if crop else None,
                "growth_stage": body.crop_stage,
                "soil_type": farmer.get("soil_type"),
                "soil_ph": farmer.get("soil_ph"),
                "soil_moisture": farmer.get("soil_moisture"),
                "current_season": season["current_season"],
                "upcoming_season": season["upcoming_season"],
            },
            "advisory": advisory_payload,
            "farmer_action_plan": _build_action_plan(alerts, weather, language, crop),
            "data_quality": {
                "weather_data_timestamp": weather.get("retrieved_at_utc"),
                "soil_data_timestamp": None,
                "advisory_generated_at": now.isoformat(),
                "missing_information": [
                    name for name, value in (
                        ("soil_type", farmer.get("soil_type")),
                        ("soil_ph", farmer.get("soil_ph")),
                        ("soil_moisture", farmer.get("soil_moisture")),
                        ("crop_growth_stage", body.crop_stage if crop else "not_applicable"),
                    ) if value is None
                ],
                "confidence": (
                    "moderate_weather_data_only"
                    if not any((farmer.get("soil_type"), farmer.get("soil_ph"), farmer.get("soil_moisture")))
                    else "weather_and_available_farmer_data"
                ),
            },
        }

    return router
