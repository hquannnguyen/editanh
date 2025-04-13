import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

# Cấu hình thư mục
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Định dạng file cho phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Tạo thumbnail để hiển thị
        img = Image.open(file_path)
        img.thumbnail((800, 800))
        thumbnail_path = os.path.join(app.config["UPLOAD_FOLDER"], "thumbnail_" + filename)
        img.save(thumbnail_path)
        
        return render_template("index.html", 
                             uploaded_image=filename,
                             thumbnail_image="thumbnail_" + filename)
    
    return redirect(url_for("index"))

@app.route("/process", methods=["POST"])
def process_image():
    # Lấy thông số từ form
    filter_type = request.form.get("filter_type")
    brightness = float(request.form.get("brightness", 1))
    contrast = float(request.form.get("contrast", 1))
    sharpness = float(request.form.get("sharpness", 1))
    rotate_angle = int(request.form.get("rotation", 0))
    flip_direction = request.form.get("flip", "none")
    grayscale = request.form.get("grayscale", "off")
    blur = request.form.get("blur", "off")
    edge_detection = request.form.get("edge_detection", "off")
    face_detection = request.form.get("face_detection", "off")
    output_format = request.form.get("output_format", "original")

    filename = request.form.get("uploaded_image")
    if not filename:
        return redirect(url_for("index"))

    # Đường dẫn file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    # Mở ảnh
    img = Image.open(file_path)
    
    # Áp dụng các hiệu chỉnh cơ bản
    if brightness != 1:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    if contrast != 1:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    if sharpness != 1:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
    
    # Chuyển đổi sang OpenCV format
    img_cv = np.array(img)
    
    # Chuyển đổi định dạng ảnh nếu cần
    if len(img_cv.shape) == 2:  # Grayscale
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    elif img_cv.shape[2] == 4:   # RGBA
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    
    # Áp dụng các bộ lọc
    if grayscale == "on":
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    if blur == "on":
        img_cv = cv2.GaussianBlur(img_cv, (15, 15), 0)
    
    if edge_detection == "on":
        img_cv = cv2.Canny(img_cv, 100, 200) if len(img_cv.shape) == 2 else cv2.Canny(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), 100, 200)
    
    # Nhận diện khuôn mặt
    if face_detection == "on":
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Xoay ảnh
    if rotate_angle != 0:
        if rotate_angle == 90:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == 180:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_180)
        elif rotate_angle == 270:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Lật ảnh
    if flip_direction != "none":
        if flip_direction == "horizontal":
            img_cv = cv2.flip(img_cv, 1)
        elif flip_direction == "vertical":
            img_cv = cv2.flip(img_cv, 0)
    
    # Lưu ảnh đã xử lý
    processed_filename = "processed." + (output_format if output_format != "original" else filename.split('.')[-1])
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
    
    # Chuyển đổi định dạng nếu cần
    if output_format != "original":
        img_pil = Image.fromarray(img_cv)
        if output_format == "jpg":
            img_pil = img_pil.convert("RGB")
        img_pil.save(processed_path)
    else:
        cv2.imwrite(processed_path, img_cv)
    
    return render_template("index.html", 
                         uploaded_image=filename,
                         thumbnail_image="thumbnail_" + filename,
                         processed_image=processed_filename)

@app.route("/crop", methods=["POST"])
def crop_image():
    # Lấy tọa độ crop từ form
    x = int(request.form.get("x"))
    y = int(request.form.get("y"))
    width = int(request.form.get("width"))
    height = int(request.form.get("height"))
    filename = request.form.get("uploaded_image")

    # Mở ảnh và crop
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img = Image.open(file_path)
    cropped_img = img.crop((x, y, x + width, y + height))

    # Lưu ảnh đã crop
    cropped_filename = "cropped_" + filename
    cropped_path = os.path.join(app.config["UPLOAD_FOLDER"], cropped_filename)
    cropped_img.save(cropped_path)

    # Tạo thumbnail mới
    cropped_img.thumbnail((800, 800))
    thumbnail_filename = "thumbnail_" + cropped_filename
    thumbnail_path = os.path.join(app.config["UPLOAD_FOLDER"], thumbnail_filename)
    cropped_img.save(thumbnail_path)

    return render_template("index.html", 
                         uploaded_image=cropped_filename,
                         thumbnail_image=thumbnail_filename)

@app.route("/download")
def download():
    processed_filename = request.args.get("filename")
    if not processed_filename:
        return redirect(url_for("index"))
    
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
    if os.path.exists(processed_path):
        return send_file(processed_path, as_attachment=True)
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Lấy PORT từ biến môi trường
    app.run(host="0.0.0.0", port=port)