"""Entry point: python -m ai_governance"""

import os
import uvicorn

from .api import app

uvicorn.run(
    app,
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8080")),
)
