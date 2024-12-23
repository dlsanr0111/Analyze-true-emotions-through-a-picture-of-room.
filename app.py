from flask import Flask, render_template, request
from ultralytics import YOLO
from diffusers import StableDiffusionPipeline
import openai
import os
import torch

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 방문자 수 파일 경로
VISITOR_COUNT_FILE = "visitor_count.txt"

# YOLO 및 Stable Diffusion 설정
yolo_model = YOLO("yolov5s.pt")
stable_diffusion_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
stable_diffusion_model.to("cuda" if torch.cuda.is_available() else "cpu")
openai.api_key = "your API Key"

# 방문자 수 관리 함수
def get_visitor_count():
    """방문자 수를 파일에서 읽거나 초기화"""
    if not os.path.exists(VISITOR_COUNT_FILE):
        with open(VISITOR_COUNT_FILE, "w") as f:
            f.write("0")
    with open(VISITOR_COUNT_FILE, "r") as f:
        return int(f.read())
def increment_visitor_count():
    count = get_visitor_count() + 1
    with open(VISITOR_COUNT_FILE, "w") as f:
        f.write(str(count))
    return count

#YoLo 탐지
def detect_objects(image_path):
    results = yolo_model(image_path)
    detected_objects = []
    for box in results[0].boxes:
        label = int(box.cls[0])
        detected_objects.append(yolo_model.names[label])
    return detected_objects

def determine_emotion_with_gpt(detected_objects):
    objects = ", ".join(detected_objects)
    prompt = (
        f"The room contains the following objects: {objects}. "
        "Based on these objects, determine which one of the 7 universal emotions the room conveys. "
        "The 7 universal emotions are: happiness, sadness, anger, fear, surprise, disgust, and contempt. "
        "Respond in the following format: 'Emotion: [Emotion], Prompt: [Water-color painting description]'."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response['choices'][0]['message']['content'].strip()

    if "Prompt:" in content:
        emotion, painting_description = content.split("Prompt:")
        emotion = emotion.replace("Emotion:", "").strip()
        painting_description = painting_description.strip()
        return emotion, painting_description
    else:
        return "unknown", "A watercolor painting with unknown emotion."


@app.route("/", methods=["GET", "POST"])
def index():
    visitor_count = increment_visitor_count()  # 방문자 수 증가
    if request.method == "POST":
        uploaded_file = request.files["image"]
        if uploaded_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)
            detected_objects = detect_objects(image_path)
            emotion, painting_prompt = determine_emotion_with_gpt(detected_objects)
            result_image_path = "static/result.png"
            result_image = stable_diffusion_model(painting_prompt, num_inference_steps=30).images[0]
            result_image.save(result_image_path)

            return render_template(
                "result.html",
                result_image=result_image_path,
                emotion=emotion,
                visitor_count=visitor_count
            )

    return render_template("index.html", visitor_count=visitor_count)

if __name__ == "__main__":
    app.run(debug=True)
