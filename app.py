"""
MediAssist - RAG-Powered Healthcare Chatbot
100% offline — no API key needed.

Models used:
- Embeddings : sentence-transformers/all-MiniLM-L6-v2  (90MB, very fast)
- LLM        : google/flan-t5-base                     (250MB, CPU-friendly)
- Vector DB  : FAISS (in-memory, instant)
"""

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import faiss
import numpy as np
import uuid
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Medical Knowledge Base ──────────────────────────────────────────
# 25 structured medical Q&A documents
KNOWLEDGE_BASE = [
    # Fever
    "Fever means body temperature above 37.5°C or 99.5°F. Common causes are bacterial infections, viral infections, and heat exhaustion. To treat fever at home: rest, drink plenty of water, and take paracetamol or ibuprofen. Go to a doctor if fever is above 39.4°C, lasts more than 3 days, or occurs in a baby under 3 months.",

    # Headache
    "Headaches have three main types: tension headache feels like pressure around the head, migraine causes throbbing pain with nausea and sensitivity to light, cluster headache causes severe pain around one eye. Common triggers include stress, dehydration, lack of sleep, and eye strain. Treatment: drink water, rest in a dark quiet room, take paracetamol or ibuprofen.",

    # Cough
    "A dry cough produces no mucus and is usually caused by a viral infection or allergies. A productive cough with mucus usually indicates bronchitis or bacterial infection. A chronic cough lasting over 8 weeks may indicate asthma or GERD. Soothe cough with honey and warm water. See a doctor if cough lasts more than 3 weeks or has blood.",

    # Chest pain
    "Chest pain warning signs: heart attack causes crushing chest pain that spreads to the left arm or jaw with sweating and shortness of breath — this is a medical emergency, call ambulance immediately. Musculoskeletal chest pain is sharp and worsens with movement. GERD causes a burning sensation in the chest after eating.",

    # Diabetes
    "Diabetes is a condition where blood sugar levels are too high. Type 1 diabetes is autoimmune and requires insulin injections. Type 2 diabetes is lifestyle-related and is managed with diet, exercise, and oral medications. Symptoms of diabetes: frequent urination, excessive thirst, unexplained weight loss, fatigue, and blurred vision. Normal fasting blood sugar is below 100 mg/dL.",

    # Blood pressure
    "High blood pressure or hypertension is when blood pressure is consistently above 130/80 mmHg. It can cause stroke, heart attack, and kidney damage if untreated. Risk factors include obesity, high salt intake, stress, smoking, and genetics. Treatment includes DASH diet, reducing sodium, regular exercise, and medications such as ACE inhibitors or beta-blockers.",

    # Asthma
    "Asthma is a chronic lung condition causing inflammation and narrowing of airways. Symptoms include wheezing, shortness of breath, chest tightness, and coughing especially at night. Common triggers are dust, pollen, cold air, exercise, and cigarette smoke. Treatment includes bronchodilator inhalers for quick relief and corticosteroid inhalers for long-term control.",

    # Anemia
    "Anemia means low hemoglobin: below 13.5 g/dL in men or 12 g/dL in women. The most common cause is iron deficiency. Symptoms include fatigue, pale skin, dizziness, and shortness of breath. Iron-rich foods include spinach, lentils, chickpeas, red meat, and fortified cereals. Take iron supplements with vitamin C juice to improve absorption. Avoid tea and coffee with iron-rich meals.",

    # Thyroid
    "Hypothyroidism means the thyroid gland is underactive and causes fatigue, weight gain, cold intolerance, dry skin, and depression. Hyperthyroidism means overactive thyroid and causes weight loss, rapid heartbeat, anxiety, and heat intolerance. Both are diagnosed with a TSH blood test. Hypothyroidism is treated with levothyroxine tablets daily.",

    # Paracetamol
    "Paracetamol is used for mild to moderate pain and fever. Adult dose is 500mg to 1g every 4 to 6 hours, maximum 4g per day. Paracetamol is safe for most people including pregnant women when used as directed. Never exceed the recommended dose as overdose causes serious liver damage. Avoid alcohol when taking paracetamol.",

    # Ibuprofen
    "Ibuprofen is a painkiller and anti-inflammatory medicine. Adult dose is 200mg to 400mg every 4 to 6 hours, always taken with food to protect the stomach. Do not take ibuprofen if you have kidney disease, stomach ulcers, or are in late pregnancy. Ibuprofen reduces fever, pain, and swelling. It interacts with blood thinners like warfarin.",

    # Antibiotics
    "Antibiotics only treat bacterial infections, not viral infections like colds or flu. Taking antibiotics for a virus does not help and creates antibiotic resistance. Common antibiotics include amoxicillin for throat and ear infections, azithromycin for chest infections, and ciprofloxacin for urinary tract infections. Always complete the full course of antibiotics even if you feel better.",

    # First aid cuts
    "For cuts and minor wounds: apply gentle pressure to stop bleeding for 5 to 10 minutes. Clean the wound under running water with mild soap. Apply antiseptic cream and cover with a clean bandage. Change dressing daily. Seek medical help for deep wounds, wounds that do not stop bleeding, or signs of infection such as redness, warmth, swelling, or pus.",

    # First aid burns
    "For burns: immediately cool the burn under cool running water for 20 minutes. Do not use ice, butter, or toothpaste on burns as these make it worse. Cover with a clean non-fluffy material. Seek emergency care for burns larger than your palm, burns on face, hands, or genitals, chemical burns, or burns with large blisters.",

    # Stroke
    "FAST method to recognize stroke: Face drooping on one side when asked to smile, Arm weakness where one arm drifts down when both raised, Speech difficulty with slurred or strange words, Time to call emergency services immediately. Stroke is a medical emergency. Every minute without treatment kills brain cells. Do not wait to see if symptoms improve.",

    # Heart attack
    "Heart attack warning signs: crushing or squeezing chest pain, pain spreading to left arm or jaw, sweating, nausea, and shortness of breath. Some heart attacks have mild symptoms especially in women. If you suspect a heart attack call an ambulance immediately. Chewing aspirin 300mg can help while waiting for ambulance if patient is not allergic.",

    # Depression
    "Depression is a medical condition, not a personal weakness. Symptoms include persistent sadness for more than 2 weeks, loss of interest in activities, fatigue, sleep problems, appetite changes, difficulty concentrating, and thoughts of self-harm. Treatments include cognitive behavioral therapy, antidepressant medications, regular exercise, and social support. Please see a doctor if experiencing these symptoms.",

    # Anxiety
    "Anxiety disorder involves excessive worry that interferes with daily life. Physical symptoms include racing heart, sweating, trembling, shortness of breath, and dizziness. Management strategies: deep breathing exercises such as inhale for 4 counts hold for 7 exhale for 8 counts, mindfulness meditation, regular physical exercise, limiting caffeine, and cognitive behavioral therapy. Medications include SSRIs for long-term and sometimes benzodiazepines short-term.",

    # Sleep
    "Good sleep hygiene for insomnia: go to bed and wake up at the same time every day including weekends. Avoid screens and bright light 1 hour before bed. Keep bedroom cool dark and quiet. Avoid caffeine after 2pm and alcohol before sleep. Do not exercise within 2 hours of bedtime. If you cannot sleep after 20 minutes get up and do a quiet activity until sleepy.",

    # Diet
    "A balanced healthy diet should include: carbohydrates 50 to 60 percent of calories from whole grains brown rice and vegetables, protein 15 to 20 percent from chicken fish eggs legumes and dairy, healthy fats 25 to 30 percent from nuts olive oil and avocado. Limit processed foods added sugar salt and saturated fats. Drink 8 glasses of water daily.",

    # Vitamin D
    "Vitamin D deficiency affects bone health, immune function, and mood. Risk factors include limited sun exposure, dark skin, obesity, and living in northern regions. Symptoms include fatigue, bone pain, muscle weakness, and frequent infections. Food sources include fatty fish egg yolks and fortified milk. Standard supplement dose is 1000 to 2000 IU daily. Get tested with a 25-OH vitamin D blood test.",

    # Iron foods
    "Foods high in iron: red meat liver spinach lentils chickpeas kidney beans tofu fortified breakfast cereals pumpkin seeds and dark chocolate. Eat iron-rich foods with vitamin C sources like orange juice tomatoes or bell peppers to double iron absorption. Avoid drinking tea coffee or milk within 1 hour of eating iron-rich foods as these reduce absorption.",

    # Dehydration
    "Dehydration occurs when you lose more fluid than you drink. Mild symptoms: thirst, dark yellow urine, dry mouth, fatigue. Moderate symptoms: headache, dizziness, reduced urination. Severe dehydration: rapid heartbeat, confusion, sunken eyes — this is a medical emergency. Treatment: sip water or oral rehydration solution slowly. Adults need about 2 to 3 litres of fluid daily.",

    # Allergies
    "Allergic reactions range from mild to life-threatening. Mild allergy symptoms: sneezing, runny nose, itchy eyes, skin rash, hives. Treat with antihistamines like cetirizine or loratadine. Severe anaphylaxis is a medical emergency with throat swelling, difficulty breathing, low blood pressure, and rapid heartbeat. Anaphylaxis requires immediate epinephrine injection and emergency care.",

    # Back pain
    "Most back pain is muscular and improves in 2 to 6 weeks. Causes include poor posture, muscle strain, prolonged sitting, and lifting heavy objects incorrectly. Treatment: stay active with gentle walking, apply ice pack first 48 hours then heat, take ibuprofen or paracetamol, and do gentle stretching. See a doctor if back pain spreads down the leg, is associated with numbness, or follows an injury.",
]

# ── Model Loading ────────────────────────────────────────────────────
logger.info("Loading embedding model (all-MiniLM-L6-v2, ~90MB)...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Embedding model ready ✓")

logger.info("Loading language model (flan-t5-base, ~250MB)...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
logger.info("Language model ready ✓")

# ── Build FAISS Index ────────────────────────────────────────────────
logger.info("Building FAISS vector index...")
kb_embeddings = embedder.encode(KNOWLEDGE_BASE, convert_to_numpy=True, normalize_embeddings=True)
dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)   # Inner product = cosine similarity (normalized vectors)
index.add(kb_embeddings)
logger.info(f"FAISS index built: {index.ntotal} documents ✓")

# ── Session store ────────────────────────────────────────────────────
sessions = {}   # session_id -> list of {"role": "user"/"bot", "text": "..."}

# ── Core RAG Function ────────────────────────────────────────────────
def rag_answer(question: str, session_id: str) -> dict:
    start = time.time()

    # Step 1: Embed the question
    q_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)

    # Step 2: Retrieve top-3 relevant documents
    scores, indices = index.search(q_vec, k=3)
    retrieved = [KNOWLEDGE_BASE[i] for i in indices[0] if i < len(KNOWLEDGE_BASE)]
    context = " ".join(retrieved)

    # Step 3: Build chat history (last 3 turns)
    history = sessions.get(session_id, [])
    history_text = ""
    if history:
        last_turns = history[-3:]
        history_text = " ".join([f"{'Patient' if h['role']=='user' else 'Assistant'}: {h['text']}" for h in last_turns])

    # Step 4: Build prompt for flan-t5
    prompt = f"""You are MediAssist, a helpful medical information assistant.
Use the medical information below to answer the patient's question clearly and accurately.
Always recommend consulting a doctor for diagnosis and treatment.

Medical Information: {context}

{f'Previous conversation: {history_text}' if history_text else ''}

Patient Question: {question}

Answer:"""

    # Step 5: Generate answer
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = llm.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Fallback if model output is too short
    if len(answer.split()) < 5:
        if retrieved:
            answer = retrieved[0][:400]
        else:
            answer = "I don't have specific information about that. Please consult a healthcare professional."

    latency = round((time.time() - start) * 1000, 1)

    # Update session history
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({"role": "user", "text": question})
    sessions[session_id].append({"role": "bot",  "text": answer})

    return {
        "answer": answer,
        "sources_used": len(retrieved),
        "top_match_score": float(scores[0][0]),
        "latency_ms": latency
    }

# ── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/v1/chat", methods=["POST"])
def chat():
    """
    POST /api/v1/chat
    Body: { "message": "...", "session_id": "..." }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Body must contain 'message'"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400
    if len(message) > 500:
        return jsonify({"error": "Message too long (max 500 characters)"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))

    try:
        result = rag_answer(message, session_id)
        return jsonify({
            "status": "success",
            "data": {
                "session_id": session_id,
                "question": message,
                "answer": result["answer"],
                "sources_used": result["sources_used"],
                "confidence": f"{round(result['top_match_score'] * 100)}%",
                "latency_ms": result["latency_ms"],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": "Failed to generate response", "detail": str(e)}), 500

@app.route("/api/v1/session/new", methods=["POST"])
def new_session():
    sid = str(uuid.uuid4())
    sessions[sid] = []
    return jsonify({"status": "success", "session_id": sid})

@app.route("/api/v1/session/<sid>/history", methods=["GET"])
def get_history(sid):
    history = sessions.get(sid, [])
    return jsonify({"status": "success", "session_id": sid, "history": history, "turns": len(history) // 2})

@app.route("/api/v1/session/<sid>/clear", methods=["DELETE"])
def clear_session(sid):
    sessions.pop(sid, None)
    return jsonify({"status": "success", "message": "Session cleared"})

@app.route("/api/v1/search", methods=["POST"])
def semantic_search():
    """
    POST /api/v1/search
    Body: { "query": "...", "top_k": 3 }
    Direct vector search without LLM generation.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Body must contain 'query'"}), 400

    top_k = min(int(data.get("top_k", 3)), 5)
    q_vec = embedder.encode([data["query"]], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_vec, k=top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(KNOWLEDGE_BASE):
            results.append({
                "rank": rank + 1,
                "content": KNOWLEDGE_BASE[idx],
                "similarity_score": f"{round(float(score) * 100)}%"
            })

    return jsonify({"status": "success", "query": data["query"], "results": results})

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "MediAssist API v1",
        "mode": "100% Offline — No API Key Required",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm": "google/flan-t5-base",
        "vector_store": f"FAISS ({index.ntotal} documents)",
        "active_sessions": len(sessions)
    })

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MediAssist — Healthcare Chatbot")
    print("  http://localhost:5002")
    print("  100% Offline — No API key needed!")
    print("  Models: all-MiniLM-L6-v2 + flan-t5-base")
    print("="*55 + "\n")
    app.run(debug=False, port=5002, threaded=True)
