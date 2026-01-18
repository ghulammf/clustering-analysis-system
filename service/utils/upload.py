import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

upload_bp = Blueprint('upload', __name__)

UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_upload_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            return jsonify({
                "status": "Error",
                "message": "Gagal menyimpan data",
                "Error": e
            })

@upload_bp.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak diizinkan"}), 400
    
    # -- Menghapus data lama
    if os.path.exists(UPLOAD_FOLDER):
        clear_upload_folder(UPLOAD_FOLDER)

    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    filename = secure_filename(f"data_fitur{ext}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    return jsonify({
        "message": "Upload berhasil",
        "filename": filename
    }), 200
