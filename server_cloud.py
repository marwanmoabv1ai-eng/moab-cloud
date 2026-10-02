import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq


app = Flask(__name__)
CORS(app)


# ==============================
# GROQ
# ==============================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set")

client = Groq(api_key=api_key)

WHISPER_MODEL = "whisper-large-v3-turbo"
AI_MODEL = "openai/gpt-oss-20b"


# ==============================
# HOME
# ==============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "MOAB",
        "status": "running",
        "server": "cloud",
        "audio_endpoint": "/ask_audio"
    })


# ==============================
# HEALTH
# ==============================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "project": "MOAB"
    })


# ==============================
# AUDIO
# ==============================

@app.route("/ask_audio", methods=["POST"])
def ask_audio():

    temp_path = None

    try:

        print("\n==============================")
        print("MOAB CLOUD - AUDIO REQUEST")
        print("==============================")

        print("Received files:", list(request.files.keys()))

        if "audio" not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio received"
            }), 400

        audio = request.files["audio"]

        if not audio.filename:
            return jsonify({
                "success": False,
                "error": "Audio filename is empty"
            }), 400

        print("Audio:", audio.filename)
        print("Mimetype:", audio.mimetype)

        # ------------------------------
        # Save temporary WAV file
        # ------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_path = temp_file.name
            audio.save(temp_path)

        print("Temporary file:", temp_path)

        # ------------------------------
        # Speech To Text
        # ------------------------------

        with open(temp_path, "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                file=("audio.wav", audio_file, "audio/wav"),
                model=WHISPER_MODEL,
                response_format="json",
                temperature=0
            )

        question = transcription.text.strip()

        print("Question:", question)

        if not question:
            return jsonify({
                "success": False,
                "error": "Could not understand the audio"
            }), 400

        # ------------------------------
        # AI
        # ------------------------------

        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MOAB, a helpful voice assistant. "
                        "Answer naturally and clearly. "
                        "Keep answers concise."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.4,
            max_completion_tokens=512
        )

        answer = completion.choices[0].message.content

        print("Answer:", answer)

        return jsonify({
            "success": True,
            "text": question,
            "response": answer
        })

    except Exception as e:

        print("ERROR:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        # ------------------------------
        # Delete temporary file
        # ------------------------------

        if temp_path:

            try:
                os.remove(temp_path)
                print("Temporary file deleted.")
            except Exception:
                pass


# ==============================
# START
# ==============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("\n==============================")
    print("MOAB CLOUD SERVER")
    print("==============================")
    print("Port:", port)
    print("==============================\n")

    app.run(
        host="0.0.0.0",
        port=port
    )