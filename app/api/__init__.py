from fastapi import APIRouter
from app.api.endpoints import router as endpoints_router

# Main API router
api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
api_router.include_router(endpoints_router)
