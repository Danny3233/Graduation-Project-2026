from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.sign import router as sign_router

app = FastAPI(
    title="Deaf Communication System API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dòng này bắt buộc phải có.
app.include_router(sign_router)


@app.get("/")
def root():
    return {"message": "Deaf Communication System API"}


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running",
    }