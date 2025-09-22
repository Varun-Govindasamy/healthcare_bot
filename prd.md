# 📱 WhatsApp AI Healthcare Chatbot – GPT-Only Flow

## 1. Onboarding Flow (First Use)
When user first opens the chatbot in WhatsApp, the bot collects:

> ⚠️ **Important:** The chatbot will **not proceed or respond to any queries** until the user provides all required medical details.

### Demographics
- Name  
- Age  
- Gender  
- District & State (for outbreak alerts)

### Medical Profile
- Allergies  
- Medication preference → English / Ayurvedic / Home remedies  
- Existing conditions (BP, Diabetes, Asthma, etc.)  
- Current medications  

### Health Data Options
- Upload **PDF/Doc** → parsed by GPT-4o  
- Upload **photo of medical report** → GPT-4o Vision extracts structured data  
- Upload **skin image** → GPT-4o Vision diagnoses skin-related issues  

👉 If details are missing, chatbot **asks manually**.  
👉 Structured profile stored in **MongoDB**.  

---

## 2. Memory & Storage
- **MongoDB** → Long-term medical profile (age, allergies, meds, district, state, preferences)
- **SQLite** → Chat memory (short-term session memory for GPT agents)  
- **Redis** → Cache recent queries (FAQ answers)  
- **Pinecone** → Vector DB for:  
  - Global healthcare docs (WHO, MoHFW)  
  - User-uploaded medical reports (separate namespace per user)  

---

## 3. Language Handling (GPT-powered)
- **Input Processing:**  
  - GPT detects user’s input language  
  - Translates → English (internal reasoning language)  

- **Output Processing:**  
  - GPT generates response in English  
  - Translates back into user’s original language before sending  

✅ One GPT pipeline handles detection, translation, and medical reasoning.  

---

## 4. CrewAI Agents (All GPT-powered)
1. **Medical Data Agent** → Gathers onboarding medical details, parses uploaded docs/reports  
2. **RAG Agent** → Uses Pinecone + `text-embedding-3-large` to fetch context  
3. **Search Agent** → Uses **Serper API** for real-time outbreak/disease updates (district/state aware)  
4. **Vision Agent** → GPT-4o Vision for reports & skin image analysis  
5. **Conversation Agent** → GPT-4o generates empathetic, safe, personalized answers  

---

## 5. Query Pipeline
### User interaction flow:
1. User sends **text/image/file** → via WhatsApp → Twilio → FastAPI backend  
2. **GPT detects + translates** language → standardizes to English  
3. Backend fetches **user profile** (MongoDB)  
   - ⚠️ If user profile is incomplete, **chatbot will not process the query** and will prompt the user to provide missing details.  
4. **CrewAI Router** decides:  
   - RAG Agent (Pinecone)  
   - Search Agent (Serper)  
   - Vision Agent (if file/image)  
   - Conversation Agent (core reasoning)  
5. GPT generates response (personalized with age, allergies, meds, etc.)  
6. GPT translates output → back into original user language  
7. Response sent via **Twilio → WhatsApp**  

---

## 6. Example Scenarios

### 🧑‍⚕️ Case A: Fever  
**User (Tamil):** “எனக்கு காய்ச்சல் இருக்கிறது”  
- GPT → Detects Tamil → Translates → English  
- Uses profile (50 yrs, diabetic, no allergies)  
- GPT response →  
  *“Take Paracetamol 500 mg, 1 tablet morning, afternoon, evening after meals.  
  Monitor sugar. If fever lasts 3 days, consult a doctor.”*  
- Translated back → Tamil → Sent to user  

---

### 📍 Case B: Local Outbreak Query  
**User (Patna, Bihar):** “क्या डेंगू फैल रहा है?”  
- GPT detects Hindi → Translates to English  
- Search Agent (Serper) fetches outbreak news for Bihar  
- GPT response →  
  *“Yes, 500 dengue cases reported this week in Patna.  
  Use mosquito nets and avoid stagnant water.”*  
- Translated back → Hindi → Sent to user  

---

### 📄 Case C: Report Upload  
**User uploads:** `blood_test.pdf` → High cholesterol  
**User asks:** “What should I eat daily?”  
- GPT parses report → Structured summary stored in Pinecone  
- GPT uses RAG →  
  *“Since cholesterol is high, avoid fried foods.  
  Eat oats, brown rice, leafy vegetables. Limit oil to 2 tsp/day.”*  
- Translated → user’s language → Sent  

---

### 🖼️ Case D: Skin Rash Photo  
**User uploads rash image**  
- GPT-4o Vision → detects fungal infection  
- GPT response →  
  *“Looks like a mild fungal infection.  
  Apply Clotrimazole cream twice daily for 1 week.  
  Keep area dry. If it spreads, consult dermatologist.”*  
- Translated → local language → Sent  

---

## 7. Safety Layer
- Every answer ends with disclaimer:  
  ⚠️ “This is AI guidance only. Please consult a doctor for confirmation.”  
- Age-based checks (avoid adult dosages for children)  
- Critical red flags → trigger emergency alert message  

---

## 8. Final Architecture
```
[WhatsApp User]
   | (Text, Image, File)
   v
[Twilio Webhook] → [FastAPI Backend]
   |
   v
GPT (Language Detection + Translation)
   |
   v
User Profile (MongoDB) + Memory (SQLite) + Cache (Redis)
   |
   v
CrewAI Router
   ┌─────────────┬──────────────┬───────────────┬───────────────┐
   | MedicalData | RAG Agent    | Search Agent  | Vision Agent  |
   | Agent       | (Pinecone)   | (Serper+Loc)  | (GPT-4o Vision)|
   └─────────────┴──────────────┴───────────────┴───────────────┘
   |
   v
[Conversation Agent (GPT-4o)]
   |
   v
GPT Translation (English → User language)
   |
   v
[Twilio] → WhatsApp → User
```

---


-- for redis, I have pulled it from docker and running it, so use redis accordingly.

## python version i'm using is 3.12.6, so install dependencies in compatible versions of this python.