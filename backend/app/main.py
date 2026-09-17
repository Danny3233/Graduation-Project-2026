from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.sign import router as sign_router
from app.routes.feedback import router as feedback_router

app = FastAPI(
    title="Hệ thống hỗ trợ giao tiếp cho người Điếc đang chạy API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://graduation-project-2026-bgvw.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(sign_router)
app.include_router(feedback_router)

@app.get("/")
def root():
    return {
        "message": "Hệ thống hỗ trợ giao tiếp cho người Điếc đang chạy API"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ổn định",
        "message": "Backend đang chạy ổn định",
    }