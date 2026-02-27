"""
FastAPI Backend for Threat Detection System
Connects frontend to threat detection model and Supabase database
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta
import jwt
import bcrypt
import os
from dotenv import load_dotenv

# Database
from supabase import create_client, Client

# Threat Detection
import torch
import re
from transformers import BertTokenizerFast, BertForSequenceClassification
from PyPDF2 import PdfReader
import io

load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

BERT_MODEL_PATH = r"C:\Users\HP\Downloads\50K BERT MODEL\domain_exapnsion"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================================
# THREAT DETECTION RULES (Embedded)
# =====================================================================

class DirectThreatDetector:
    """Rule-based detector for direct threat statements"""
    
    def __init__(self):
        self.benign_keywords = [
            'file', 'report', 'submit', 'deliver', 'send', 'email', 'mail',
            'meet', 'meeting', 'appointment', 'conference', 'call',
            'work', 'task', 'project', 'assignment', 'deadline',
            'class', 'lecture', 'presentation', 'training', 'seminar',
            'document', 'paper', 'proposal', 'thesis', 'essay',
            'interview', 'hiring', 'recruitment', 'application',
            'travel', 'flight', 'trip', 'visit', 'tour',
            'party', 'event', 'celebration', 'gathering', 'ceremony',
            'research', 'study', 'analysis', 'investigation', 'experiment',
            'maintenance', 'repair', 'installation', 'construction',
            'shopping', 'purchase', 'buying', 'selling', 'trading',
            'exercise', 'workout', 'training', 'practice', 'sport',
            'performance', 'show', 'concert', 'exhibition', 'display',
            'medical', 'doctor', 'hospital', 'appointment', 'treatment',
            'review', 'audit', 'inspection', 'evaluation', 'assessment',
            'chicken', 'duck', 'goose', 'pig', 'cow', 'beef', 'pork', 'lamb', 'turkey', 'deer', 'fish', 'animal', 'hunt', 'hunting', 'butcher', 'meat', 'cook', 'cooking', 'dinner', 'lunch', 'meal', 'food', 'recipe', 'slaughterhouse',
            'game', 'video game', 'minecraft', 'fortnite', 'call of duty', 'enemy', 'monster', 'zombie', 'quest', 'boss', 'level', 'character', 'npc', 'player', 'achievement', 'movie', 'film', 'scene', 'actor', 'villain', 'hero', 'play', 'playing',
        ]
        
        self.intent_patterns = [
            r'\b(i|we)\s+(will|am\s+going\s+to|plan\s+to|intend\s+to|am\s+planning|shall).{0,200}(kill|attack|bomb|shoot|stab|poison|harm|destroy|target|hit|blast|explode|detonate|ram|crash|slaughter|massacre|murder)\b',
            r'\b(i|we)\s+(want|need|have)\s+to.{0,200}(kill|attack|bomb|shoot|stab|poison|harm|destroy|target|hit|blast|explode|detonate|ram|crash|slaughter|massacre|murder)\b',
            r'\b(i|we)\s+(kill|attack|bomb|shoot|stab|poison|harm|destroy|will\s+harm|will\s+kill)\b',
        ]
        
        self.action_patterns = [
            r'\b(ram|crash|drive|collide)\s+(a|an|my|the|his|her|their|our)?\s*(vehicle|car|truck|bus)\s+(into|through|toward)\b',
            r'\b(blow|detonate|set off|trigger)\s+(a|an|the)?\s*(bomb|explosive|device|c4|explosives)\b',
            r'\b(shoot|fire|open fire|release)\s+(at|on)\s+(the|a|an)?\s*(crowd|people|building|target|civilians)\b',
            r'\b(knife|stab|slash|cut)\s+(people|civilians|targets|someone)\b',
            r'\b(poison|gas|infect)\s+(the|a|an)?\s*(water|food|air|crowd|people)\b',
        ]
        
        self.temporal_patterns = [
            r'\b(tomorrow|tonight|this\s+(week|weekend|month)|next\s+(week|month|year)|at\s+\d{1,2}:\d{2}|on\s+\d{1,2}/\d{1,2})\b',
            r'\b(before|after|at|around)\s+(dawn|dusk|midnight|noon|morning|afternoon|evening)\b',
        ]
        
        self.target_patterns = [
            r'\b(target|targets|hit|attack)\s+(the|a|an)?\s*(crowd|people|building|event|concert|school|mall|market|station|airport|government|police|military)\b',
            r'\b(at|near|inside)\s+(the|a|an)?\s*(event|location|building|stadium|church|mosque|synagogue|school|market|station)\b',
        ]
        
        self.cyber_threat_patterns = [
            r'\b(failure|fail|refuse|don\'t|do not)\s+(to\s+)?(meet|comply|pay|give|provide)\s+(our|my|the)?\s*(demands|terms|requirements|conditions)\s+(will|shall|would)\s+(result in|lead to|cause)\s+(the\s+)?(release|exposure|publication|leak|disclosure|distribution)\s+(of)?\s*(sensitive|private|personal|confidential|explicit|compromising|intimate)?\s*(information|data|photos|files|documents|details)\b',
            r'\b(release|expose|publish|leak|disclose|share|distribute|post)\s+(sensitive|private|personal|confidential|explicit|compromising)\s+(information|data|photos|files|documents)\s+(if|unless|until)\s+(you|your|they)\s+(don\'t|do not|refuse|fail)\b',
            r'\b(pay|give|provide|transfer)\s+(money|funds|bitcoin|ransom|payment|fee)\s+(or|else|otherwise)\s+(we\s+)?(will|shall|would)\s+(release|expose|leak|publicize)\b',
            r'\b(have|obtained|stolen|compromised|breached|hacked|accessed)\s+(your|the)?\s*(password|credentials|account|system|database|server|network|email)\b',
            r'\b(will|shall|would)\s+(attack|breach|hack|compromise|take down|crash|destroy|lock|encrypt|delete)\s+(your|the)?\s*(system|server|network|website|database|data|files)\b',
            r'\b(DDoS|ddos attack|launch.*attack|flood.*network)\b',
        ]
    
    def check_benign_context(self, text):
        text_lower = text.lower()
        benign_count = sum(1 for keyword in self.benign_keywords if keyword in text_lower)
        return benign_count >= 2
    
    def check_intent(self, text):
        text_lower = text.lower()
        if self.check_benign_context(text):
            return False, None
        if re.search(r'\b(i|we|me|us)\b', text_lower):
            harmful_verbs = ['kill', 'attack', 'bomb', 'shoot', 'stab', 'poison', 'harm', 'destroy', 'target', 'hit', 'blast', 'explode', 'detonate', 'ram', 'crash', 'slaughter', 'massacre', 'murder']
            for verb in harmful_verbs:
                if re.search(r'\b' + verb + r'\b', text_lower):
                    return True, "intent"
        for pattern in self.intent_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "intent"
        return False, None
    
    def check_action(self, text):
        text_lower = text.lower()
        for pattern in self.action_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "specific_action"
        return False, None
    
    def check_temporal(self, text):
        text_lower = text.lower()
        for pattern in self.temporal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "temporal"
        return False, None
    
    def check_target(self, text):
        text_lower = text.lower()
        for pattern in self.target_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "target"
        return False, None
    
    def check_cyber(self, text):
        text_lower = text.lower()
        for pattern in self.cyber_threat_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "cyber_threat"
        return False, None
    
    def detect_direct_threat(self, text):
        if self.check_benign_context(text):
            return False, 0.0, []
        
        has_cyber, cyber_type = self.check_cyber(text)
        if has_cyber:
            return True, 0.92, ["cyber_threat"]
        
        has_intent, intent_type = self.check_intent(text)
        if has_intent:
            return True, 0.93, ["intent"]
        
        indicators = []
        has_action, action_type = self.check_action(text)
        if has_action:
            indicators.append("specific_action")
        
        has_temporal, temporal_type = self.check_temporal(text)
        if has_temporal:
            indicators.append("temporal")
        
        has_target, target_type = self.check_target(text)
        if has_target:
            indicators.append("target")
        
        if len(indicators) >= 2:
            return True, 0.95, indicators
        elif len(indicators) == 1 and has_action:
            return True, 0.90, indicators
        else:
            return False, 0.0, []

# Initialize threat detector
rule_detector = DirectThreatDetector()

# Load BERT model
print("Loading BERT model...")
try:
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    tokenizer = None
    model = None

# =====================================================================
# FASTAPI APP SETUP
# =====================================================================

app = FastAPI(title="Threat Detection API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# PYDANTIC MODELS
# =====================================================================

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class PredictionRequest(BaseModel):
    input_text: str
    input_type: str = "text"  # text, pdf, txt, audio

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    total_predictions: int
    total_threats_detected: int

class ThreatLogResponse(BaseModel):
    id: int
    created_at: str
    input_text: str
    input_type: str
    prediction: str
    confidence: float
    detection_method: str
    threat_indicators: str

# =====================================================================
# AUTHENTICATION HELPERS
# =====================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(password: str, hash_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode(), hash_password.encode())

def create_access_token(user_id: str) -> str:
    """Create JWT token"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: str) -> str:
    """Dependency to get current user from token"""
    payload = verify_token(token)
    return payload.get("user_id")

# =====================================================================
# AUTHENTICATION ENDPOINTS
# =====================================================================

@app.post("/api/auth/signup", response_model=TokenResponse)
async def sign_up(request: SignUpRequest):
    """Register new user"""
    try:
        # Check if user exists
        existing = supabase.table('users').select('id').eq('email', request.email).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create user
        password_hash = hash_password(request.password)
        response = supabase.table('users').insert({
            'email': request.email,
            'password_hash': password_hash,
            'full_name': request.full_name,
            'created_at': datetime.now().isoformat()
        }).execute()
        
        user_id = response.data[0]['id']
        
        # Create subscription
        supabase.table('subscriptions').insert({
            'user_id': user_id,
            'plan': 'free',
            'predictions_limit': 100
        }).execute()
        
        # Create token
        token = create_access_token(user_id)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/signin", response_model=TokenResponse)
async def sign_in(request: SignInRequest):
    """Login user"""
    try:
        response = supabase.table('users').select('*').eq('email', request.email).execute()
        
        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user = response.data[0]
        
        if not verify_password(request.password, user['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        supabase.table('users').update({
            'last_login': datetime.now().isoformat()
        }).eq('id', user['id']).execute()
        
        # Create token
        token = create_access_token(user['id'])
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user['id']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# USER ENDPOINTS
# =====================================================================

@app.get("/api/user/profile", response_model=UserResponse)
async def get_profile(token: str):
    """Get user profile"""
    try:
        user_id = get_current_user(token)
        response = supabase.table('users').select('*').eq('id', user_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = response.data[0]
        return {
            "id": user['id'],
            "email": user['email'],
            "full_name": user['full_name'],
            "total_predictions": user['total_predictions'],
            "total_threats_detected": user['total_threats_detected']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# PREDICTION ENDPOINTS
# =====================================================================

@app.post("/api/predict")
async def predict(request: PredictionRequest, token: str):
    """Make a threat prediction"""
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        user_id = get_current_user(token)
        text = request.input_text
        
        # Rule-based detection
        is_rule_threat, rule_score, threat_reasons = rule_detector.detect_direct_threat(text)
        
        if is_rule_threat:
            prediction = "threatening"
            confidence = rule_score
            detection_method = "rule-based"
        else:
            # BERT prediction
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            threat_prob = probabilities[0][1].item()
            
            if threat_prob >= 0.80:
                prediction = "threatening"
                confidence = threat_prob
            else:
                prediction = "non_threatening"
                confidence = 1 - threat_prob
            
            detection_method = "bert-model"
            threat_reasons = []
        
        # Log to database
        log_response = supabase.table('threat_logs').insert({
            'user_id': user_id,
            'input_text': text[:1000],
            'input_type': request.input_type,
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'detection_method': detection_method,
            'threat_indicators': str(threat_reasons),
            'created_at': datetime.now().isoformat()
        }).execute()
        
        # Update user stats
        user = supabase.table('users').select('*').eq('id', user_id).execute().data[0]
        threat_count = user['total_threats_detected']
        if prediction == 'threatening':
            threat_count += 1
        
        supabase.table('users').update({
            'total_predictions': user['total_predictions'] + 1,
            'total_threats_detected': threat_count
        }).eq('id', user_id).execute()
        
        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "detection_method": detection_method,
            "threat_indicators": threat_reasons,
            "log_id": log_response.data[0]['id']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# HISTORY ENDPOINTS
# =====================================================================

@app.get("/api/history")
async def get_history(
    token: str,
    input_type: Optional[str] = None,
    prediction: Optional[str] = None,
    limit: int = 50
):
    """Get user's threat log history with optional filters"""
    try:
        user_id = get_current_user(token)
        
        query = supabase.table('threat_logs').select('*').eq('user_id', user_id)
        
        if input_type:
            query = query.eq('input_type', input_type)
        
        if prediction:
            query = query.eq('prediction', prediction)
        
        response = query.order('created_at', desc=True).limit(limit).execute()
        
        return {
            "success": True,
            "logs": response.data,
            "count": len(response.data)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics(token: str):
    """Get user's statistics by input type"""
    try:
        user_id = get_current_user(token)
        response = supabase.table('threat_logs').select('*').eq('user_id', user_id).execute()
        
        stats = {
            'text': {'total': 0, 'threatening': 0},
            'pdf': {'total': 0, 'threatening': 0},
            'txt': {'total': 0, 'threatening': 0},
            'audio': {'total': 0, 'threatening': 0}
        }
        
        for log in response.data:
            input_type = log['input_type']
            if input_type in stats:
                stats[input_type]['total'] += 1
                if log['prediction'] == 'threatening':
                    stats[input_type]['threatening'] += 1
        
        return {"success": True, "statistics": stats}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history/{log_id}")
async def delete_log(log_id: int, token: str):
    """Delete a log entry"""
    try:
        user_id = get_current_user(token)
        
        # Verify ownership
        log_response = supabase.table('threat_logs').select('*').eq('id', log_id).execute()
        if not log_response.data or log_response.data[0]['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        supabase.table('threat_logs').delete().eq('id', log_id).execute()
        
        return {"success": True, "message": "Log deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# HEALTH CHECK
# =====================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "database_connected": supabase is not None
    }

# =====================================================================
# RUN SERVER
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
