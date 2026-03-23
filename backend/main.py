import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from recommender import (
    content_based_recommend,
    personalized_recommend,
    hybrid_recommend,
    get_all_products,
    get_all_customers,
    get_all_categories,
    build_content_recommender,
    build_user_recommender,
)
from analytics import (
    get_kpi_metrics,
    get_top_products,
    get_geo_sales,
    get_sales_timeline,
    get_customer_stats,
)

app = FastAPI(
    title="Product Recommendation API",
    description="ML-powered product recommendation engine for e-commerce",
    version="1.0.0",
)

# Accept localhost in dev + the Vercel frontend URL set via env var on Render
_frontend_url = os.getenv("FRONTEND_URL", "")
_allowed_origins = ["http://localhost:3000"]
if _frontend_url:
    _allowed_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"https://(.*\.vercel\.app|.*\.onrender\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- STARTUP ----------------------

@app.on_event("startup")
async def startup_event():
    """Pre-warm ML models on startup so first request is fast."""
    print("Pre-warming recommendation models...")
    build_content_recommender()
    build_user_recommender()
    print("Models ready.")


# ---------------------- SCHEMAS ----------------------

class ContentRecommendRequest(BaseModel):
    product_name: str
    num_recommendations: int = 5


class CollaborativeRecommendRequest(BaseModel):
    customer_id: str
    num_recommendations: int = 5


class HybridRecommendRequest(BaseModel):
    product_name: str
    customer_id: Optional[str] = None
    num_recommendations: int = 5
    content_weight: float = 0.6
    collab_weight: float = 0.4


# ---------------------- HEALTH ----------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------- PRODUCTS & CUSTOMERS ----------------------

@app.get("/api/products")
def list_products(category: str = Query(default=None)):
    """Return product names, optionally filtered by category."""
    return {"products": get_all_products(category)}


@app.get("/api/categories")
def list_categories():
    """Return all product categories with their product counts."""
    return {"categories": get_all_categories()}


@app.get("/api/customers")
def list_customers():
    """Return all valid customer IDs."""
    return {"customers": get_all_customers()}


# ---------------------- RECOMMENDATIONS ----------------------

@app.post("/api/recommend/content")
def recommend_content(req: ContentRecommendRequest):
    """Content-based recommendations using TF-IDF + FAISS similarity."""
    results = content_based_recommend(req.product_name, req.num_recommendations)
    if results is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"product": req.product_name, "type": "content", "recommendations": results}


@app.post("/api/recommend/collaborative")
def recommend_collaborative(req: CollaborativeRecommendRequest):
    """Collaborative filtering recommendations based on similar customer purchases."""
    try:
        customer_id = float(req.customer_id)
    except ValueError:
        customer_id = req.customer_id

    results = personalized_recommend(customer_id, req.num_recommendations)
    return {"customerId": req.customer_id, "type": "collaborative", "recommendations": results}


@app.post("/api/recommend/hybrid")
def recommend_hybrid(req: HybridRecommendRequest):
    """Hybrid recommendations combining content-based and collaborative filtering."""
    try:
        customer_id = float(req.customer_id) if req.customer_id else None
    except ValueError:
        customer_id = req.customer_id

    results = hybrid_recommend(
        req.product_name,
        customer_id,
        req.num_recommendations,
        req.content_weight,
        req.collab_weight,
    )
    return {
        "product": req.product_name,
        "customerId": req.customer_id,
        "type": "hybrid",
        "weights": {"content": req.content_weight, "collaborative": req.collab_weight},
        "recommendations": results,
    }


# ---------------------- ANALYTICS ----------------------

@app.get("/api/analytics/kpi")
def analytics_kpi():
    """Overall KPI metrics: revenue, orders, customers, etc."""
    return get_kpi_metrics()


@app.get("/api/analytics/top-products")
def analytics_top_products(n: int = Query(default=10, ge=1, le=50)):
    """Top N products by revenue and by quantity sold."""
    return get_top_products(n)


@app.get("/api/analytics/geo-sales")
def analytics_geo_sales(product: str = Query(default=None)):
    """Sales breakdown by country, optionally filtered by product description."""
    return {"data": get_geo_sales(product)}


@app.get("/api/analytics/timeline")
def analytics_timeline(product: str = Query(default=None)):
    """Monthly revenue and order count over time, optionally filtered by product description."""
    return {"data": get_sales_timeline(product)}


@app.get("/api/analytics/customers")
def analytics_customers():
    """Customer stats: totals, top spenders, distribution by country."""
    return get_customer_stats()


# ---------------------- ENTRY POINT ----------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
