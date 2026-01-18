from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import os
import uuid
import shutil
from pathlib import Path
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio
from minio import Minio
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
import stripe

from facefindr.facefindr import FaceFindr

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FaceFindr API")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKKET_NAME")
MINIO_SSL = os.getenv("MINIO_SSL", "false").lower() == "true"
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "https://minio.facemenow.co")

# Stripe configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

stripe.api_key = STRIPE_SECRET_KEY

print("MINIO_ENDPOINT:", MINIO_ENDPOINT)
print("MINIO_ACCESS_KEY:", MINIO_ACCESS_KEY)
print("MINIO_SECRET_KEY:", MINIO_SECRET_KEY)
print("STRIPE_SECRET_KEY:", "***" if STRIPE_SECRET_KEY else "Not set")

db_url = os.getenv("DATABASE_URL") 
face_finder = FaceFindr(db_path=db_url)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Separate executors for search and upload to prevent blocking
search_executor = ThreadPoolExecutor(max_workers=4)  # Priority for search - fast response
upload_executor = ThreadPoolExecutor(max_workers=2)  # Background upload processing

# -------------------- PYDANTIC MODELS --------------------

class EventCreate(BaseModel):
    event_name: str
    event_details: str
    event_img_url: Optional[str] = None
    start_at: datetime
    end_at: datetime

class CUserCreate(BaseModel):
    email: str
    password: str
    img_url: Optional[str] = None
    # User details
    province: Optional[str] = None
    code: Optional[int] = None
    bank_name: Optional[str] = None
    bank_id: Optional[str] = None
    phone: Optional[str] = None
    bank_copy_img_url: Optional[str] = None
    line_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    show_name: Optional[str] = None
    set_details: Optional[str] = None
    can_paid: Optional[bool] = True

class DUserCreate(BaseModel):
    email: str
    display_name: str
    display_details: Optional[str] = None
    profile_url: Optional[str] = None
    consent: Optional[bool] = True

class DonateRequest(BaseModel):
    event_id: str
    duser_id: str
    donate_price: float
    currency: str = "thb"  # Default to Thai Baht
    return_url: Optional[str] = None

class StripePaymentIntent(BaseModel):
    email: str
    event_id: str
    duser_id: str
    amount: float  
    currency: str = "thb"
    user_details: Optional[str] = None
    automatic_payment_methods: bool = True
    return_url: Optional[str] = None

class CartItem(BaseModel):
    buser_id: str
    img_id: str

# -------------------- DATABASE UTILITIES --------------------

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(db_url)

def execute_query(query: str, params: tuple = None, fetch: bool = False):
    """Execute database query"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                conn.commit()
                return cursor.rowcount
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# -------------------- EXISTING UTILITIES --------------------

async def process_image_for_search_async(img_url: str, tolerance: float = 0.4, max_results: int = 30) -> Dict:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        search_executor,
        face_finder.search_image_from_url_formatted,
        img_url,
        tolerance,
        max_results
    )

async def process_image_for_search_local_async(image_path: str, tolerance: float = 0.4, max_results: int = 30, event_id: str = None) -> Dict:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        search_executor,
        face_finder.search_image_from_local_file_formatted,
        image_path,
        tolerance,
        max_results,
        event_id
    )

    grouped_results = {
        "exact_matches": [],
        "high_matches": [],
        "partial_matches": []
    }

    for face_data, confidence in results:
        match_info = {
            "confidence": float(confidence),
            "face_id": face_data["face_id"],
            "img_id": face_data["img_id"],
            "face_location": face_data["face_location"],
            "face_create_at": face_data["face_create_at"],
            "image": face_data["image"],
            "user": face_data["user"],
            "event": face_data["event"]
        }

        if confidence >= 0.99:
            grouped_results["exact_matches"].append(match_info)
        elif confidence >= 0.95:
            grouped_results["high_matches"].append(match_info)
        else:
            grouped_results["partial_matches"].append(match_info)

    for group in grouped_results.values():
        group.sort(key=lambda x: x["confidence"], reverse=True)

    stats = {
        "total_matches": len(results),
        "exact_matches": len(grouped_results["exact_matches"]),
        "high_matches": len(grouped_results["high_matches"]),
        "partial_matches": len(grouped_results["partial_matches"])
    }

    return {
        "statistics": stats,
        "matches": grouped_results
    }
    
def upload_to_minio(file_path: Path, object_name: str) -> str:
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SSL
    )
    bucket = BUCKET_NAME
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    client.fput_object(bucket, object_name, str(file_path))
    return f"{MINIO_PUBLIC_URL}/{bucket}/{object_name}"

async def add_image_to_db_async(image_path: str, img_url: str, cuser_id: str, event_id: str) -> Dict:
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        upload_executor,
        face_finder.process_image_from_url,
        img_url,
        cuser_id,
        event_id
    )
    
    return {
        "faces_found": len(result), 
        "id": result[0].img_id if result else None,
        "faces": result
    }

# -------------------- EVENT --------------------

@app.post("/events")
async def create_event(
    event_name: str = Form(...),
    event_details: str = Form(...),
    start_at: datetime = Form(...),
    end_at: datetime = Form(...),
    event_img: UploadFile = File(...)
):
    event_id = str(uuid.uuid4())
    current_time = datetime.now()

    unique_name = f"{event_id}_{event_img.filename}"
    temp_path = UPLOAD_DIR / unique_name
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(event_img.file, buffer)

    event_img_url = upload_to_minio(temp_path, unique_name)
    
    query = """
        INSERT INTO events (id, event_name, event_details, event_img_url, start_at, end_at, create_at, update_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    execute_query(query, (
        event_id, event_name, event_details, event_img_url,
        start_at, end_at, current_time, current_time
    ))

    # 4. Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    return {
        "message": "Event created successfully",
        "event_id": event_id,
        "event_name": event_name,
        "event_img_url": event_img_url
    }

@app.get("/events")
async def get_events():
    """Get all events"""
    query = "SELECT * FROM events ORDER BY create_at DESC"
    events = execute_query(query, fetch=True)
    
    return {
        "events": [dict(event) for event in events]
    }

@app.get("/events/{event_id}")
async def get_event(event_id: str):
    """Get specific event"""
    query = "SELECT * FROM events WHERE id = %s"
    event = execute_query(query, (event_id,), fetch=True)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return dict(event[0])

# -------------------- USER MANAGEMENT --------------------

@app.post("/cusers")
async def create_cuser(cuser: CUserCreate):
    """Create a new camera user (photographer)"""
    cuser_id = str(uuid.uuid4())
    details_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    # First create user details
    details_query = """
        INSERT INTO cuser_details (id, province, code, bank_name, bank_id, phone, bank_copy_img_url, 
                                 line_id, first_name, last_name, show_name, set_details, can_paid, create_at, update_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    execute_query(details_query, (
        details_id, cuser.province, cuser.code, cuser.bank_name, cuser.bank_id, cuser.phone,
        cuser.bank_copy_img_url, cuser.line_id, cuser.first_name, cuser.last_name, cuser.show_name,
        cuser.set_details, cuser.can_paid, current_time, current_time
    ))
    
    # Then create user
    user_query = """
        INSERT INTO cusers (id, email, password, img_url, create_at, update_at, cuser_details_fk)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    execute_query(user_query, (
        cuser_id, cuser.email, cuser.password, cuser.img_url, current_time, current_time, details_id
    ))
    
    return {
        "message": "Camera user created successfully",
        "cuser_id": cuser_id,
        "email": cuser.email
    }

@app.post("/dusers")
async def create_duser(duser: DUserCreate):
    current_time = datetime.now()

    query = """
        INSERT INTO dusers (id, display_name, display_details, profile_url, consent, create_at, update_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            display_details = EXCLUDED.display_details,
            profile_url = EXCLUDED.profile_url,
            consent = EXCLUDED.consent,
            update_at = EXCLUDED.update_at
    """

    execute_query(query, (
        duser.email, 
        duser.display_name,
        duser.display_details,
        duser.profile_url,
        duser.consent,
        current_time,
        current_time
    ))

    return {
        "duser_id": duser.email,
        "display_name": duser.display_name,
        "message": "Donor user created or updated successfully"
    }


@app.get("/cusers")
async def get_cusers():
    """Get all camera users"""
    query = """
        SELECT c.*, cd.* FROM cusers c
        LEFT JOIN cuser_details cd ON c.cuser_details_fk = cd.id
        ORDER BY c.create_at DESC
    """
    cusers = execute_query(query, fetch=True)
    
    return {
        "cusers": [dict(cuser) for cuser in cusers]
    }

@app.get("/dusers")
async def get_dusers():
    """Get all donor users"""
    query = "SELECT * FROM dusers ORDER BY create_at DESC"
    dusers = execute_query(query, fetch=True)
    
    return {
        "dusers": [dict(duser) for duser in dusers]
    }

# -------------------- STRIPE PAYMENT SYSTEM --------------------

@app.post("/create-payment-intent")
async def create_payment_intent(payment_data: StripePaymentIntent):
    try:
        # Check if event exists
        event_query = "SELECT id, event_name FROM events WHERE id = %s"
        event = execute_query(event_query, (payment_data.event_id,), fetch=True)
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Check if duser exists
        duser_query = "SELECT id, display_name FROM dusers WHERE id = %s"
        duser = execute_query(duser_query, (payment_data.duser_id,), fetch=True)
        
        if not duser:
            raise HTTPException(status_code=404, detail="Donor user not found")
        
        # Convert amount to cents (Stripe requires amount in smallest currency unit)
        amount_cents = int(payment_data.amount * 100)
        
        # Create payment intent
        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency=payment_data.currency,
            metadata={
                'event_id': payment_data.event_id,
                'duser_id': payment_data.duser_id,
                'event_name': event[0]['event_name'],
                'donor_name': duser[0]['display_name'],
                'original_amount': str(payment_data.amount)
            },
            automatic_payment_methods={
                'enabled': payment_data.automatic_payment_methods
            }
        )
        
        # Store pending payment in database
        pending_payment_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        pending_query = """
            INSERT INTO pending_payments (id, stripe_payment_intent_id, event_id, duser_id, 
                                        amount, currency, status, create_at, update_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        execute_query(pending_query, (
            pending_payment_id, intent.id, payment_data.event_id, payment_data.duser_id,
            payment_data.amount, payment_data.currency, 'pending', current_time, current_time
        ))
        
        return {
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
            "amount": payment_data.amount,
            "currency": payment_data.currency,
            "event_name": event[0]['event_name'],
            "donor_name": duser[0]['display_name']
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(f"Payment intent creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Payment intent creation failed: {str(e)}")

@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook")

    # ✅ เพิ่มตรงนี้
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        await handle_checkout_session_completed(session)

    elif event['type'] == 'payment_intent.succeeded':
        payment_intent = event['data']['object']
        await handle_successful_payment(payment_intent)

    return {"status": "success"}



@app.post("/create-checkout-session")
async def create_checkout_session(payment_data: StripePaymentIntent):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["promptpay"],
            line_items=[{
                "price_data": {
                    "currency": payment_data.currency,
                    "product_data": {
                        "name": f"Thank you for your donation!",
                    },
                    "unit_amount": int(payment_data.amount * 100),
                },
                "quantity": 1,
            }],
            mode="payment",
            customer_email=payment_data.email,
            payment_intent_data={
                "metadata": {
                    "event_id": payment_data.event_id,
                    "duser_id": payment_data.duser_id,
                    "original_amount": str(payment_data.amount),
                    "email": payment_data.email,
                    "user_details": payment_data.user_details or ""
                }
            },
            success_url=payment_data.return_url + "?success=true",
            cancel_url=payment_data.return_url + "?cancelled=true",
        )


        return {"checkout_url": session.url}

    except Exception as e:
        logger.error(f"Checkout session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


async def handle_checkout_session_completed(session):
    try:
        metadata = session.get("metadata", {})
        event_id = metadata.get("event_id")
        duser_id = metadata.get("duser_id")
        user_details = metadata.get("user_details")
        amount = float(metadata.get("amount", 0))
        stripe_payment_intent_id = session.get("payment_intent")

        if not event_id or not duser_id:
            logger.warning("Missing metadata in session")
            return

        donation_id = str(uuid.uuid4())
        current_time = datetime.now()

        query = """
            INSERT INTO leaderboard (id, event_id, duser_id, donate_price, stripe_payment_intent_id,
                                     payment_status, user_details, create_at, update_at)
            VALUES (%s, %s, %s, %s, %s, 'completed', %s, %s, %s)
        """
        execute_query(query, (
            donation_id, event_id, duser_id, amount,
            stripe_payment_intent_id, user_details, current_time, current_time
        ))

        logger.info(f"[Checkout] Donation saved: {donation_id}")

    except Exception as e:
        logger.error(f"Failed to handle checkout session: {e}")


async def handle_successful_payment(payment_intent):
    """Handle successful payment confirmation"""
    try:
        # Extract metadata
        metadata = payment_intent.get('metadata', {})
        event_id = metadata.get('event_id')
        duser_id = metadata.get('duser_id')
        user_details = metadata.get("user_details")
        original_amount = float(metadata.get('original_amount', 0))
        
        if not event_id or not duser_id:
            logger.error(f"Missing metadata in payment intent: {payment_intent['id']}")
            return
        
        # Create donation record
        donation_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        donation_query = """
            INSERT INTO leaderboard (id, event_id, duser_id, donate_price, stripe_payment_intent_id, 
                                   payment_status, user_details, create_at, update_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        execute_query(donation_query, (
            donation_id, event_id, duser_id, original_amount, payment_intent['id'],
            'completed', user_details, current_time, current_time
        ))
        
        # Update pending payment status
        update_pending_query = """
            UPDATE pending_payments 
            SET status = 'completed', update_at = %s 
            WHERE stripe_payment_intent_id = %s
        """
        
        execute_query(update_pending_query, (current_time, payment_intent['id']))
        
        logger.info(f"Donation completed: {donation_id} for amount {original_amount}")
        
    except Exception as e:
        logger.error(f"Error handling successful payment: {e}")

async def handle_failed_payment(payment_intent):
    """Handle failed payment"""
    try:
        current_time = datetime.now()
        
        # Update pending payment status
        update_pending_query = """
            UPDATE pending_payments 
            SET status = 'failed', update_at = %s 
            WHERE stripe_payment_intent_id = %s
        """
        
        execute_query(update_pending_query, (current_time, payment_intent['id']))
        
        logger.info(f"Payment failed: {payment_intent['id']}")
        
    except Exception as e:
        logger.error(f"Error handling failed payment: {e}")

@app.get("/payment-status/{payment_intent_id}")
async def get_payment_status(payment_intent_id: str):
    """Check payment status"""
    try:
        # Check in our database first
        query = """
            SELECT pp.*, e.event_name, d.display_name 
            FROM pending_payments pp
            LEFT JOIN events e ON pp.event_id = e.id
            LEFT JOIN dusers d ON pp.duser_id = d.id
            WHERE pp.stripe_payment_intent_id = %s
        """
        
        payment = execute_query(query, (payment_intent_id,), fetch=True)
        
        if not payment:
            raise HTTPException(status_code=404, detail="Payment not found")
        
        payment_data = dict(payment[0])
        
        # Also check with Stripe
        try:
            stripe_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
            payment_data['stripe_status'] = stripe_intent.status
        except stripe.error.StripeError as e:
            logger.warning(f"Could not retrieve Stripe payment intent: {e}")
            payment_data['stripe_status'] = 'unknown'
        
        return payment_data
        
    except Exception as e:
        logger.error(f"Error checking payment status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking payment status: {str(e)}")

# -------------------- DONATION SYSTEM (UPDATED) --------------------

@app.post("/donate")
async def donate_to_event(donation: DonateRequest):
    """Create a donation (now redirects to Stripe payment)"""
    
    payment_data = StripePaymentIntent(
        event_id=donation.event_id,
        duser_id=donation.duser_id,
        amount=donation.donate_price,
        currency=donation.currency,
        return_url=donation.return_url
    )
    
    return await create_payment_intent(payment_data)

@app.post("/donate-direct")
async def donate_direct(donation: DonateRequest):
    """Direct donation without payment processing (for testing/admin)"""
    donation_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    event_query = "SELECT id FROM events WHERE id = %s"
    event = execute_query(event_query, (donation.event_id,), fetch=True)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    duser_query = "SELECT id FROM dusers WHERE id = %s"
    duser = execute_query(duser_query, (donation.duser_id,), fetch=True)
    
    if not duser:
        raise HTTPException(status_code=404, detail="Donor user not found")
    
    query = """
        INSERT INTO leaderboard (id, event_id, duser_id, donate_price, payment_status, create_at, update_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    execute_query(query, (
        donation_id, donation.event_id, donation.duser_id, donation.donate_price, 
        'direct', current_time, current_time
    ))
    
    return {
        "message": "Direct donation successful",
        "donation_id": donation_id,
        "amount": donation.donate_price
    }

@app.get("/events/{event_id}/leaderboard")
async def get_event_leaderboard(
    event_id: str,
    sort_by: str = Query("amount", enum=["amount", "recent"]),
    limit: int = Query(50, ge=1, le=100)
):

    event_query = "SELECT id FROM events WHERE id = %s"
    event = execute_query(event_query, (event_id,), fetch=True)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if sort_by == "amount":
        order_clause = "ORDER BY total_donated DESC, MAX(l.create_at) DESC"
    else: 
        order_clause = "ORDER BY MAX(l.create_at) DESC"
    
    query = f"""
        SELECT 
            d.id as duser_id,
            d.display_name,
            d.display_details,
            d.profile_url,
            SUM(l.donate_price) as total_donated,
            COUNT(l.id) as donation_count,
            MAX(l.create_at) as last_donation_at,
            COUNT(CASE WHEN l.payment_status = 'completed' THEN 1 END) as completed_payments,
            COUNT(CASE WHEN l.payment_status = 'direct' THEN 1 END) as direct_payments,
            STRING_AGG(l.user_details, ', ' ORDER BY l.create_at DESC) as user_details
        FROM leaderboard l
        JOIN dusers d ON l.duser_id = d.id
        WHERE l.event_id = %s AND l.payment_status IN ('completed', 'direct')
        GROUP BY d.id, d.display_name, d.display_details, d.profile_url
        {order_clause}
        LIMIT %s
    """
    
    leaderboard = execute_query(query, (event_id, limit), fetch=True)
    
    stats_query = """
        SELECT 
            COUNT(DISTINCT l.duser_id) as total_donors,
            SUM(l.donate_price) as total_amount,
            COUNT(l.id) as total_donations,
            COUNT(CASE WHEN l.payment_status = 'completed' THEN 1 END) as stripe_donations,
            COUNT(CASE WHEN l.payment_status = 'direct' THEN 1 END) as direct_donations
        FROM leaderboard l
        WHERE l.event_id = %s AND l.payment_status IN ('completed', 'direct')
    """
    
    stats = execute_query(stats_query, (event_id,), fetch=True)
    
    return {
        "event_id": event_id,
        "sort_by": sort_by,
        "statistics": dict(stats[0]) if stats else {},
        "leaderboard": [dict(entry) for entry in leaderboard]
    }

# -------------------- IMAGE CART SYSTEM --------------------

@app.post("/cart/add")
async def add_to_cart(item: CartItem):
    """Add an image to user's cart"""
    cart_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    # Check if item already exists in cart
    check_query = "SELECT id FROM images_cart WHERE buser_id = %s AND img_id = %s"
    existing = execute_query(check_query, (item.buser_id, item.img_id), fetch=True)
    
    if existing:
        raise HTTPException(status_code=400, detail="Item already in cart")
    
    # Check if image exists
    img_query = "SELECT id FROM images WHERE id = %s"
    image = execute_query(img_query, (item.img_id,), fetch=True)
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Add to cart
    query = """
        INSERT INTO images_cart (id, buser_id, img_id, create_at, update_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    
    execute_query(query, (cart_id, item.buser_id, item.img_id, current_time, current_time))
    
    return {
        "message": "Item added to cart successfully",
        "cart_id": cart_id
    }

@app.delete("/cart/remove")
async def remove_from_cart(buser_id: str, img_id: str):
    """Remove an image from user's cart"""
    query = "DELETE FROM images_cart WHERE buser_id = %s AND img_id = %s"
    
    rows_affected = execute_query(query, (buser_id, img_id))
    
    if rows_affected == 0:
        raise HTTPException(status_code=404, detail="Item not found in cart")
    
    return {
        "message": "Item removed from cart successfully"
    }

@app.get("/cart/{buser_id}")
async def get_user_cart(buser_id: str):
    """Get user's cart contents"""
    query = """
        SELECT 
            ic.id as cart_id,
            ic.create_at as added_at,
            i.id as img_id,
            i.img_url,
            i.file_hash,
            i.metadata,
            i.create_at as image_created_at,
            e.id as event_id,
            e.event_name,
            c.id as cuser_id,
            cd.show_name as photographer_name
        FROM images_cart ic
        JOIN images i ON ic.img_id = i.id
        LEFT JOIN events e ON i.event_id = e.id
        LEFT JOIN cusers c ON i.cuser_id = c.id
        LEFT JOIN cuser_details cd ON c.cuser_details_fk = cd.id
        WHERE ic.buser_id = %s
        ORDER BY ic.create_at DESC
    """
    
    cart_items = execute_query(query, (buser_id,), fetch=True)
    
    return {
        "buser_id": buser_id,
        "cart_items": [dict(item) for item in cart_items]
    }

# -------------------- CUSER STATISTICS --------------------

@app.get("/cusers/{cuser_id}/events")
async def get_cuser_events(cuser_id: str):
    """Get events where cuser has uploaded images"""
    query = """
        SELECT 
            e.id as event_id,
            e.event_name,
            e.event_details,
            e.start_at,
            e.end_at,
            COUNT(i.id) as image_count,
            MIN(i.create_at) as first_upload,
            MAX(i.create_at) as last_upload
        FROM events e
        JOIN images i ON e.id = i.event_id
        WHERE i.cuser_id = %s
        GROUP BY e.id, e.event_name, e.event_details, e.start_at, e.end_at
        ORDER BY last_upload DESC
    """
    
    events = execute_query(query, (cuser_id,), fetch=True)
    
    return {
        "cuser_id": cuser_id,
        "events": [dict(event) for event in events]
    }

@app.get("/cusers/{cuser_id}/images")
async def get_cuser_images(cuser_id: str, event_id: Optional[str] = None):
    """Get images uploaded by cuser, optionally filtered by event"""
    base_query = """
        SELECT 
            i.id as img_id,
            i.img_url,
            i.file_hash,
            i.processed,
            i.metadata,
            i.create_at,
            e.id as event_id,
            e.event_name,
            COUNT(f.id) as face_count
        FROM images i
        LEFT JOIN events e ON i.event_id = e.id
        LEFT JOIN faces f ON i.id = f.img_id
        WHERE i.cuser_id = %s
    """
    
    params = [cuser_id]
    
    if event_id:
        base_query += " AND i.event_id = %s"
        params.append(event_id)
    
    base_query += """
        GROUP BY i.id, i.img_url, i.file_hash, i.processed, i.metadata, i.create_at, e.id, e.event_name
        ORDER BY i.create_at DESC
    """
    
    images = execute_query(base_query, tuple(params), fetch=True)
    
    return {
        "cuser_id": cuser_id,
        "event_id": event_id,
        "images": [dict(image) for image in images]
    }

# -------------------- DUSER STATISTICS --------------------

@app.get("/dusers/{duser_id}/downloads")
async def get_duser_downloads(duser_id: str):
    """Get download statistics for duser by event"""
    query = """
        SELECT 
            e.id as event_id,
            e.event_name,
            e.event_details,
            COUNT(ic.id) as images_in_cart,
            MIN(ic.create_at) as first_added,
            MAX(ic.create_at) as last_added
        FROM events e
        JOIN images i ON e.id = i.event_id
        JOIN images_cart ic ON i.id = ic.img_id
        WHERE ic.buser_id = %s
        GROUP BY e.id, e.event_name, e.event_details
        ORDER BY last_added DESC
    """
    
    downloads = execute_query(query, (duser_id,), fetch=True)
    
    return {
        "duser_id": duser_id,
        "downloads_by_event": [dict(download) for download in downloads]
    }

# -------------------- MODIFIED UPLOAD ENDPOINT --------------------

@app.post("/upload-to-db")
async def upload_image_to_database(
    file: UploadFile = File(...),
    cuser_id: str = Form(...),
    event_id: str = Form(...)
):
    """Upload image to database with event association"""
    temp_path = None
    try:
        # Check if cuser exists
        cuser_query = "SELECT id FROM cusers WHERE id = %s"
        cuser = execute_query(cuser_query, (cuser_id,), fetch=True)
        
        if not cuser:
            raise HTTPException(status_code=404, detail="Camera user not found")
        
        # Check if event exists
        event_query = "SELECT id FROM events WHERE id = %s"
        event = execute_query(event_query, (event_id,), fetch=True)
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Record the upload in history
        history_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        history_query = """
            INSERT INTO cuser_event_history (id, cuser_id, event_id, create_at, update_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        execute_query(history_query, (history_id, cuser_id, event_id, current_time, current_time))
        
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        temp_path = UPLOAD_DIR / unique_name

        # Save the file temporarily
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to MinIO and get the image URL
        img_url = upload_to_minio(temp_path, unique_name)  

        # Process and save to database
        db_add_result = await add_image_to_db_async(str(temp_path), img_url, cuser_id, event_id)

        return {
            "faces_found": db_add_result["faces_found"],
            "img_url": img_url,
            "id": db_add_result["id"],
            "event_id": event_id,
            "history_id": history_id
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_path}: {e}")

# -------------------- EXISTING ENDPOINTS --------------------

@app.post("/search-image")
async def search_uploaded_image(file: UploadFile = File(...), event_id: str = None):
    temp_path = None
    try:
        # 1. Create unique file name
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        temp_path = UPLOAD_DIR / unique_name

        # 2. Save the file temporarily
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Search for matches using local file path (don't upload to MinIO)
        results = await process_image_for_search_local_async(str(temp_path), event_id=event_id)

        return {
            "message": f"Search complete. Found {results['statistics']['total_matches']} total matches.",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search image: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_path}: {e}")

@app.get("/search-image-url")
async def search_image_by_url_get(img_url: str):
    """
    Search for faces in an image using a URL (GET method for easier testing)
    """
    try:
        results = await process_image_for_search_async(img_url)
        return {
            "message": f"Search complete. Found {results['statistics']['total_matches']} total matches.",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search image: {str(e)}")

@app.get("/debug-faces")
async def debug_faces():
    """
    Debug endpoint to check face data in database
    """
    try:
        debug_info = face_finder.face_db.debug_face_data()
        return debug_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-corrupted-faces")
async def clear_corrupted_faces():
    """
    Clear corrupted face embeddings from database
    """
    try:
        result = face_finder.face_db.clear_corrupted_faces()
        return {
            "message": "Corrupted faces cleared successfully",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/find-corrupted-embeddings")
async def find_corrupted_embeddings():
    """
    Find and report corrupted embeddings without deleting them
    """
    try:
        result = face_finder.face_db.find_corrupted_embeddings()
        return {
            "message": "Corrupted embeddings analysis complete",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)