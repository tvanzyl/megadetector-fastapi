import asyncio
import logging
import time
from typing import Optional

from config import DETECTOR_METADATA
from fastapi import FastAPI
from handlers.images import load_image
from handlers.models import get_megadetector_model
from pydantic import BaseModel, Field, validator

# Setup logger & environment variables
logger = logging.getLogger(__name__)

# Add Security
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Create a FastAPI app
app = FastAPI()

# Add Security
security = HTTPBasic()

def get_current_credentials(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"smart"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"savannahs"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


# Pydantic model for the request body
# -------------------------------------------------------------------------------------------
AVAILABLE_MODELS = list(i[0] for i in DETECTOR_METADATA.keys())
AVAILABLE_VERSIONS = list(i[1] for i in DETECTOR_METADATA.keys())

class AnnotateRequest(BaseModel):
    image: str = Field(
        default="https://upload.wikimedia.org/wikipedia/commons/7/74/Grevy%27s_Zebra_Stallion.jpg",
        description="Web url or a base64 encoded image to annotate",
    )
    model_name: str = Field(AVAILABLE_MODELS[0], description="name of model to use")
    model_version: str = Field(AVAILABLE_VERSIONS[0], description="Version of model to use")
    detection_threshold: Optional[float] = Field(default=0.2, description="Optional Detection threshold")

    @validator("model_name")
    def name_validator(cls, name):
        if name not in AVAILABLE_MODELS:
            raise ValueError(f"Invalid model name: {name}. Please choose from {AVAILABLE_MODELS}")
        return name
    
    @validator("model_version")
    def version_validator(cls, version):
        if version not in AVAILABLE_VERSIONS:
            raise ValueError(f"Invalid model version: {version}. Please choose from {AVAILABLE_VERSIONS}")
        return version


# API endpoints
# -------------------------------------------------------------------------------------------
@app.post("/annotate/")
async def get_annotated_image(credentials: Annotated[str, Depends(get_current_credentials)], request: AnnotateRequest):
    """Main function that consumes an image list object and returns a list of annotations"""
    # Load the megadetector model. To keep the memory footprint low, keep only one model at any time
    start = time.perf_counter()
    image, model = await asyncio.gather(
        load_image(request.image),
        get_megadetector_model((request.model_name,request.model_version)),
    )

    # Set the detection threshold if provided
    if request.detection_threshold is None:
        detection_threshold = model.detection_threshold
    else:
        detection_threshold = request.detection_threshold

    annotation = model.annotate_image(image, request.image[:255], detection_threshold)
    annotation_time = time.perf_counter() - start

    return {
        "annotation": annotation,
        "model_version": request.model_version,
        "model_name": request.model_name,
        "detection_threshold": detection_threshold,
        "annotation_time": annotation_time
    }

@app.get("/available_models/")
def get_available_models():
    """Returns a list of available models & its metadata"""
    tmp = {"available_models": {i+"_"+j:DETECTOR_METADATA[(i,j)] for i,j in zip(AVAILABLE_MODELS,AVAILABLE_VERSIONS)}}

    return tmp


@app.get("/")
def home():
    return "I'm alive & healthy! To interact with me, go to /docs for Swagger UI or /redoc for ReDoc."

