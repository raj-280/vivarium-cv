from fastapi import FastAPI
from api.routes import infer, levels, cages, alerts
from api.middleware import register_middleware

app = FastAPI(title="Vivarium CV API", version="0.1.0")
register_middleware(app)
app.include_router(infer.router)
app.include_router(levels.router)
app.include_router(cages.router)
app.include_router(alerts.router)