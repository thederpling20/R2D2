from fastapi import FastAPI
from fastapi import File, UploadFile
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable


app = FastAPI()
        
@app.post("/upload")

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path
