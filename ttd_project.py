
import os
import torch
import random
import numpy as np
from pydub import AudioSegment
from PyPDF2 import PdfReader
from transformers import BertTokenizerFast, BertForSequenceClassification
import speech_recognition as sr
import re
from supabase import create_client, Client
from datetime import datetime

# -------------------------
# SUPABASE SETUP
# -------------------------
SUPABASE_URL = "https://uhfqdzalcnastdmtlkch.supabase.co"  
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVoZnFkemFsY25hc3RkbXRsa2NoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIxMDEyNDMsImV4cCI6MjA4NzY3NzI0M30.C-nsKu7neJbQtDJL923TEzc8OlDqTAw6DnRbgh-EbYc"  # Replace with your anon key

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase_connected = True
    print("✓ Connected to Supabase")
except Exception as e:
    supabase_connected = False
    print(f"✗ Supabase connection failed: {e}")

def log_threat_to_supabase(input_text, input_type, prediction, confidence, detection_method, threat_indicators):
    """
    Log threat detection to Supabase database
    
    Args:
        input_text (str): The text that was analyzed
        input_type (str): Type of input (text, pdf, audio, txt)
        prediction (str): threatening or non_threatening
        confidence (float): Confidence percentage (0-100)
        detection_method (str): The detection method used (e.g., 'rule-based', 'bert-model')
        threat_indicators (list): List of threat indicators detected
    """
    if not supabase_connected:
        print("⚠ Supabase not connected, skipping log")
        return False
    
    try:
        # Prepare data
        log_entry = {
            'input_text': input_text[:1000],  # Truncate if too long
            'input_type': input_type,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'detection_method': detection_method,
            'threat_indicators': str(threat_indicators),  # Convert list to string
            'model_version': '1.0'
        }
        
        # Insert into database
        response = supabase.table('threat_logs').insert(log_entry).execute()
        
        print("✓ Logged to Supabase")
        return True
    
    except Exception as e:
        print(f"✗ Error logging to Supabase: {e}")
        return False

# -------------------------
# THREAT DETECTION RULES (Built-in - No external import needed)
# -------------------------
class DirectThreatDetector:
    """
    Rule-based detector for direct threat statements.
    Catches obvious threats that BERT might miss.
    Includes false positive filtering for benign activities.
    """
    
    def __init__(self):
        # Benign activity keywords - activities that are NOT threats
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
            # Animal/hunting/food related (these legitimize "kill")
            'chicken', 'duck', 'goose', 'pig', 'cow', 'beef', 'pork', 'lamb', 'turkey', 'deer', 'fish', 'animal', 'hunt', 'hunting', 'butcher', 'meat', 'cook', 'cooking', 'dinner', 'lunch', 'meal', 'food', 'recipe', 'slaughterhouse',
            # Video game/movie related (these legitimize "kill/attack/destroy")
            'game', 'video game', 'minecraft', 'fortnite', 'call of duty', 'enemy', 'monster', 'zombie', 'quest', 'boss', 'level', 'character', 'npc', 'player', 'achievement', 'movie', 'film', 'scene', 'actor', 'villain', 'hero', 'play', 'playing',
        ]
        
        # Intent patterns - very flexible to catch ANY threat statement
        self.intent_patterns = [
            # "i/we will [anything] kill/attack/etc [anything]" - no word limit
            r'\b(i|we)\s+(will|am\s+going\s+to|plan\s+to|intend\s+to|am\s+planning|shall).{0,200}(kill|attack|bomb|shoot|stab|poison|harm|destroy|target|hit|blast|explode|detonate|ram|crash|slaughter|massacre|murder)\b',
            # "I want/need to [anything] kill/etc"
            r'\b(i|we)\s+(want|need|have)\s+to.{0,200}(kill|attack|bomb|shoot|stab|poison|harm|destroy|target|hit|blast|explode|detonate|ram|crash|slaughter|massacre|murder)\b',
            # Direct "I kill/attack/will harm"
            r'\b(i|we)\s+(kill|attack|bomb|shoot|stab|poison|harm|destroy|will\s+harm|will\s+kill)\b',
        ]
        
        # Capability markers
        self.capability_patterns = [
            r'\b(i|we)\s+(have|got|obtained|possess|can|could)\s+(access|obtained|gotten|a|an)?\s*(bomb|gun|weapon|explosives|knife|vehicle|chemical|biological)\b',
            r'\b(i|we)\s+(know|can)\s+(how to)\s+(make|create|build|detonate)\b',
            r'\b(can\s+)?access\s+(the|a|an)?\s*(building|location|facility|target|crowd)\b',
        ]
        
        # Specific harmful actions
        self.action_patterns = [
            r'\b(ram|crash|drive|collide)\s+(a|an|my|the|his|her|their|our)?\s*(vehicle|car|truck|bus)\s+(into|through|toward)\b',
            r'\b(blow|detonate|set off|trigger)\s+(a|an|the)?\s*(bomb|explosive|device|c4|explosives)\b',
            r'\b(shoot|fire|open fire|release)\s+(at|on)\s+(the|a|an)?\s*(crowd|people|building|target|civilians)\b',
            r'\b(knife|stab|slash|cut)\s+(people|civilians|targets|someone)\b',
            r'\b(poison|gas|infect)\s+(the|a|an)?\s*(water|food|air|crowd|people)\b',
        ]
        
        # Temporal specificity
        self.temporal_patterns = [
            r'\b(tomorrow|tonight|this\s+(week|weekend|month)|next\s+(week|month|year)|at\s+\d{1,2}:\d{2}|on\s+\d{1,2}/\d{1,2})\b',
            r'\b(before|after|at|around)\s+(dawn|dusk|midnight|noon|morning|afternoon|evening)\b',
        ]
        
        # Target specification
        self.target_patterns = [
            r'\b(target|targets|hit|attack)\s+(the|a|an)?\s*(crowd|people|building|event|concert|school|mall|market|station|airport|government|police|military)\b',
            r'\b(at|near|inside)\s+(the|a|an)?\s*(event|location|building|stadium|church|mosque|synagogue|school|market|station)\b',
        ]
        
        # Benign patterns
        self.benign_patterns = [
            r'will\s+(file|submit|send|deliver|report|mail|email|upload)',
            r'will\s+(meet|call|contact|reach out)',
            r'will\s+(attend|join|participate|present|speak)',
            r'will\s+(work|complete|finish|accomplish|fulfill)',
            r'will\s+(arrive|depart|leave|travel)',
            r'(morning|afternoon|evening|dawn|dusk|night)\s+(meeting|appointment|class|shift|work)',
            r'will\s+(review|analyze|study|research|investigate)',
            r'i\s+(am|will be)\s+(going to|attending|participating)',
        ]
        
        # Cyber/Extortion threat patterns
        self.cyber_threat_patterns = [
            # Extortion/blackmail
            r'\b(failure|fail|refuse|don\'t|do not)\s+(to\s+)?(meet|comply|pay|give|provide)\s+(our|my|the)?\s*(demands|terms|requirements|conditions)\s+(will|shall|would)\s+(result in|lead to|cause)\s+(the\s+)?(release|exposure|publication|leak|disclosure|distribution)\s+(of)?\s*(sensitive|private|personal|confidential|explicit|compromising|intimate)?\s*(information|data|photos|files|documents|details)\b',
            r'\b(release|expose|publish|leak|disclose|share|distribute|post)\s+(sensitive|private|personal|confidential|explicit|compromising)\s+(information|data|photos|files|documents)\s+(if|unless|until)\s+(you|your|they)\s+(don\'t|do not|refuse|fail)\b',
            # Ransom/demands
            r'\b(pay|give|provide|transfer)\s+(money|funds|bitcoin|ransom|payment|fee)\s+(or|else|otherwise)\s+(we\s+)?(will|shall|would)\s+(release|expose|leak|publicize)\b',
            # Hacking/credential theft
            r'\b(have|obtained|stolen|compromised|breached|hacked|accessed)\s+(your|the)?\s*(password|credentials|account|system|database|server|network|email)\b',
            # Threat to disrupt/destroy systems
            r'\b(will|shall|would)\s+(attack|breach|hack|compromise|take down|crash|destroy|lock|encrypt|delete)\s+(your|the)?\s*(system|server|network|website|database|data|files)\b',
            # DDoS/system threat
            r'\b(DDoS|ddos attack|launch.*attack|flood.*network)\b',
        ]
    
    def check_benign_context(self, text):
        """Check if text is about benign activities (false positive filter)"""
        text_lower = text.lower()
        benign_count = sum(1 for keyword in self.benign_keywords if keyword in text_lower)
        if benign_count >= 2:
            return True
        for pattern in self.benign_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def check_intent(self, text):
        """Check for explicit intent to harm (excluding benign contexts)"""
        text_lower = text.lower()
        
        # Quick benign check first
        if self.check_benign_context(text):
            return False, None
        
        # Check if first-person statement with harmful intent
        if re.search(r'\b(i|we|me|us)\b', text_lower):
            # Check if any harmful verbs are mentioned
            harmful_verbs = ['kill', 'attack', 'bomb', 'shoot', 'stab', 'poison', 'harm', 'destroy', 'target', 'hit', 'blast', 'explode', 'detonate', 'ram', 'crash', 'slaughter', 'massacre', 'murder']
            for verb in harmful_verbs:
                if re.search(r'\b' + verb + r'\b', text_lower):
                    return True, "intent"
        
        # Also check explicit patterns
        for pattern in self.intent_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "intent"
        return False, None
    
    def check_capability(self, text):
        """Check for stated capability/access"""
        text_lower = text.lower()
        for pattern in self.capability_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "capability"
        return False, None
    
    def check_specific_action(self, text):
        """Check for specific harmful actions"""
        text_lower = text.lower()
        for pattern in self.action_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "specific_action"
        return False, None
    
    def check_temporal_specificity(self, text):
        """Check for specific timing"""
        text_lower = text.lower()
        for pattern in self.temporal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "temporal"
        return False, None
    
    def check_target_specificity(self, text):
        """Check for specific targets"""
        text_lower = text.lower()
        for pattern in self.target_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "target"
        return False, None
    
    def check_cyber_threat(self, text):
        """Check for cyber threats (extortion, hacking, DDoS, etc)"""
        text_lower = text.lower()
        for pattern in self.cyber_threat_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "cyber_threat"
        return False, None
    
    def detect_direct_threat(self, text, required_indicators=2):
        """Detect direct threats by checking for multiple indicators"""
        
        # FIRST: Check if this is about benign activities
        if self.check_benign_context(text):
            return False, 0.0, []
        
        # CHECK FOR CYBER THREATS (single indicator is enough for cyber threats)
        has_cyber, cyber_type = self.check_cyber_threat(text)
        if has_cyber:
            return True, 0.92, ["cyber_threat"]
        
        # CHECK FOR INTENT (first-person + harmful verb)
        # Intent alone should be enough for a threat
        has_intent, intent_type = self.check_intent(text)
        if has_intent:
            return True, 0.93, ["intent"]
        
        threat_indicators = []
        threat_types = []
        
        has_capability, cap_type = self.check_capability(text)
        if has_capability:
            threat_indicators.append("capability")
            threat_types.append(cap_type)
        
        has_action, action_type = self.check_specific_action(text)
        if has_action:
            threat_indicators.append("specific_action")
            threat_types.append(action_type)
        
        has_temporal, temp_type = self.check_temporal_specificity(text)
        if has_temporal:
            threat_indicators.append("temporal")
            threat_types.append(temp_type)
        
        has_target, target_type = self.check_target_specificity(text)
        if has_target:
            threat_indicators.append("target")
            threat_types.append(target_type)
        
        num_indicators = len(threat_indicators)
        
        if num_indicators >= required_indicators:
            threat_score = 0.95
            is_threat = True
        elif num_indicators == required_indicators - 1 and has_action:
            threat_score = 0.90
            is_threat = True
        else:
            threat_score = 0.0
            is_threat = False
        
        return is_threat, threat_score, threat_types

# -------------------------
# Config (optimized)
# -------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256

# Initialize rule-based detector
rule_detector = DirectThreatDetector()

# -------------------------
# PDF Text Extraction Function
# -------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a given PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while processing PDF {pdf_path}: {e}")
        return ""
    return text

# -------------------------
# TXT Text Extraction Function
# -------------------------
def extract_text_from_txt(txt_path):
    """Extracts text content from a given TXT file."""
    text = ""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: TXT file not found at {txt_path}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while processing TXT {txt_path}: {e}")
        return ""
    return text

# -------------------------
# Audio Transcription Function
# -------------------------
def transcribe_audio(audio_path):
    """Transcribes audio content from a given audio file to text."""
    text = ""
    r = sr.Recognizer()
    temp_wav_path = "temp_audio.wav"
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export(temp_wav_path, format="wav")

        with sr.AudioFile(temp_wav_path) as source:
            audio_listened = r.record(source)
            text = r.google_recognize(audio_listened)

    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_path}")
        text = ""
    except Exception as e:
        print(f"An unexpected error occurred during audio transcription: {e}")
        text = ""
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
    return text

# -------------------------
# Text Chunking Function
# -------------------------
def chunk_text(text, tokenizer, max_len):
    """Tokenizes the input text and splits it into chunks of a specified maximum length."""
    tokenized_input = tokenizer(text, return_offsets_mapping=True, truncation=False)
    input_ids = tokenized_input['input_ids']
    offset_mapping = tokenized_input['offset_mapping']

    chunks = []
    chunk_content_len = max_len - 2

    for i in range(0, len(input_ids), chunk_content_len):
        current_chunk_ids = input_ids[i : i + chunk_content_len]
        chunk_with_special_tokens = [tokenizer.cls_token_id] + current_chunk_ids + [tokenizer.sep_token_id]
        current_chunk_offset_mapping = offset_mapping[i : i + chunk_content_len]

        if current_chunk_offset_mapping:
            start_char = current_chunk_offset_mapping[0][0]
            end_char = current_chunk_offset_mapping[-1][1]
            text_segment = text[start_char:end_char]
            chunks.append(text_segment)

    return chunks

# -------------------------
# HYBRID PREDICTION FUNCTION (Rule-based + BERT)
# -------------------------
def predict_text_hybrid(text, tokenizer, model, threshold=0.80, rule_detector=None):
    """
    Predicts threat classification using both rule-based and BERT model.
    Rule-based layer runs first to catch obvious direct threats.
    
    Args:
        text (str): The input text.
        tokenizer: The pre-trained tokenizer.
        model: The trained BERT model.
        threshold (float): Probability threshold for BERT's 'threatening' class.
        rule_detector: DirectThreatDetector instance.
    
    Returns:
        tuple: (predicted_label_index, probability, detection_method, threat_reasons)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # STEP 1: Rule-based detection (catches obvious direct threats)
    if rule_detector:
        is_rule_threat, rule_score, threat_reasons = rule_detector.detect_direct_threat(
            text, 
            required_indicators=2  # Requires 2+ threat indicators
        )
        
        if is_rule_threat:
            # High-confidence direct threat detected
            print(f"[Rule-Based Detection] Direct threat indicators found: {threat_reasons}")
            return 1, rule_score, "rule-based", threat_reasons
    
    # STEP 2: BERT model prediction (for contextual threats)
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    threat_prob = probabilities[0][1].item()
    non_threat_prob = probabilities[0][0].item()

    if threat_prob >= threshold:
        predicted_label_index = 1  # Threatening
        probability = threat_prob
    else:
        predicted_label_index = 0  # Non-threatening
        probability = non_threat_prob

    return predicted_label_index, probability, "bert-model", None

# -------------------------
# Unified Input Processing for Prediction (with hybrid detection)
# -------------------------
def process_input_for_prediction(input_data, input_type, tokenizer, model, rule_detector=None):
    """
    Processes various input types and returns prediction using hybrid approach.
    """
    text = ""
    if input_type == 'text':
        text = input_data
    elif input_type == 'pdf':
        text = extract_text_from_pdf(input_data)
    elif input_type == 'txt':
        text = extract_text_from_txt(input_data)
    elif input_type == 'audio':
        text = transcribe_audio(input_data)
    else:
        print(f"Error: Unknown input type: {input_type}")
        return None, None, None, None

    if not text:
        print("Error: No text extracted/transcribed.")
        return None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Check if text needs chunking
    tokenized_full_text = tokenizer(text, truncation=False, return_tensors="pt")
    if tokenized_full_text['input_ids'].shape[1] > MAX_LEN:
        text_chunks = chunk_text(text, tokenizer, MAX_LEN)
        all_predictions = []
        all_probabilities = []
        all_methods = []
        all_reasons = []

        for chunk in text_chunks:
            # Use hybrid prediction for each chunk
            predicted_label, probability, method, reasons = predict_text_hybrid(
                chunk, tokenizer, model, rule_detector=rule_detector
            )
            all_predictions.append(predicted_label)
            all_probabilities.append(probability)
            all_methods.append(method)
            all_reasons.append(reasons)

        # If ANY chunk is flagged as threat by rules, return threat
        for i, method in enumerate(all_methods):
            if method == "rule-based" and all_predictions[i] == 1:
                return 1, all_probabilities[i], "rule-based", all_reasons[i]
        
        # Otherwise, use BERT's highest threat probability
        if all_probabilities:
            max_idx = all_probabilities.index(max(all_probabilities))
            return (
                all_predictions[max_idx],
                all_probabilities[max_idx],
                all_methods[max_idx],
                all_reasons[max_idx]
            )
        else:
            print("Error: No chunks processed for long text.")
            return None, None, None, None
    else:
        # Use hybrid prediction for non-chunked text
        return predict_text_hybrid(text, tokenizer, model, rule_detector=rule_detector)


# -------------------------
# Main interactive loop
# -------------------------
print("Loading trained model and tokenizer...")
LOADED_MODEL_PATH = r"C:\Users\HP\Downloads\50K BERT MODEL\domain_exapnsion"
try:
    inference_tokenizer = BertTokenizerFast.from_pretrained(LOADED_MODEL_PATH, local_files_only=True)
    inference_model = BertForSequenceClassification.from_pretrained(LOADED_MODEL_PATH, local_files_only=True)
    print("Trained model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading trained model or tokenizer: {e}")
    inference_tokenizer = None
    inference_model = None

if inference_model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model.to(device)
    inference_model.eval()
    print(f"Model moved to {device}.")
    print("Rule-based threat detector initialized.")


while True:
    input_choice = input("How would you like to provide input? Type '1' for text, '2' for file upload (audio/PDF/TXT), or 'q' to quit: ")

    if input_choice.lower() == 'q':
        print("Exiting program.")
        break

    if not inference_model or not inference_tokenizer:
        print("Cannot perform prediction: Model or tokenizer not loaded.")
        continue

    prediction_result = None

    if input_choice == '1':  # Text input
        text_input = input("Please enter your text for threat detection (or type 'b' to go back): ")
        if text_input.lower() == 'b':
            continue
        predicted_label_index, probability, method, reasons = process_input_for_prediction(
            text_input, 'text', inference_tokenizer, inference_model, rule_detector
        )
        if predicted_label_index is not None:
            prediction_result = {
                'predicted_label': 'threatening' if predicted_label_index == 1 else 'non_threatening',
                'confidence': probability,
                'detection_method': method,
                'threat_reasons': reasons
            }

    elif input_choice == '2':  # File upload
        file_type_choice = input("Enter '1' for a .txt file, '2' for an audio file, '3' for a PDF file, or 'b' to go back: ")
        if file_type_choice.lower() == 'b':
            continue

        if file_type_choice == '1':  # TXT file
            txt_file = input("Please provide the path to your TXT file (or type 'b' to go back): ")
            if txt_file.lower() == 'b':
                continue
            txt_file = txt_file.strip('"').strip("' ")
            print(f"Processing TXT file: {txt_file}")
            predicted_label_index, probability, method, reasons = process_input_for_prediction(
                txt_file, 'txt', inference_tokenizer, inference_model, rule_detector
            )
            if predicted_label_index is not None:
                prediction_result = {
                    'predicted_label': 'threatening' if predicted_label_index == 1 else 'non_threatening',
                    'confidence': probability,
                    'detection_method': method,
                    'threat_reasons': reasons
                }

        elif file_type_choice == '2':  # Audio file
            audio_file = input("Please provide the path to your audio file (or type 'b' to go back): ")
            if audio_file.lower() == 'b':
                continue
            audio_file = audio_file.strip('"').strip("' ")
            print(f"Processing audio file: {audio_file}")
            predicted_label_index, probability, method, reasons = process_input_for_prediction(
                audio_file, 'audio', inference_tokenizer, inference_model, rule_detector
            )
            if predicted_label_index is not None:
                prediction_result = {
                    'predicted_label': 'threatening' if predicted_label_index == 1 else 'non_threatening',
                    'confidence': probability,
                    'detection_method': method,
                    'threat_reasons': reasons
                }

        elif file_type_choice == '3':  # PDF file
            pdf_file = input("Please provide the path to your PDF file (or type 'b' to go back): ")
            if pdf_file.lower() == 'b':
                continue
            pdf_file = pdf_file.strip('"').strip("' ")
            print(f"Processing PDF file: {pdf_file}")
            predicted_label_index, probability, method, reasons = process_input_for_prediction(
                pdf_file, 'pdf', inference_tokenizer, inference_model, rule_detector
            )
            if predicted_label_index is not None:
                prediction_result = {
                    'predicted_label': 'threatening' if predicted_label_index == 1 else 'non_threatening',
                    'confidence': probability,
                    'detection_method': method,
                    'threat_reasons': reasons
                }
        else:
            print("Invalid file type choice. Returning to main menu.")
            continue

    else:
        print("Invalid choice. Please type '1', '2', or 'q'.")
        continue

    if prediction_result:
        # Convert confidence to percentage
        confidence_percentage = prediction_result['confidence'] * 100
        label = prediction_result['predicted_label'].upper()
        
        print(f"\n{'='*60}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence_percentage:.2f}%")
        print(f"Detection Method: {prediction_result['detection_method']}")
        if prediction_result['threat_reasons']:
            print(f"Threat Indicators: {prediction_result['threat_reasons']}")
        print(f"{'='*60}\n")
        
        # NEW: Log to Supabase
        log_threat_to_supabase(
            input_text=text_input if input_choice == '1' else pdf_file if file_type_choice == '3' else txt_file if file_type_choice == '1' else audio_file,
            input_type='text' if input_choice == '1' else 'pdf' if file_type_choice == '3' else 'txt' if file_type_choice == '1' else 'audio',
            prediction=prediction_result['predicted_label'],
            confidence=confidence_percentage,
            detection_method=prediction_result['detection_method'],
            threat_indicators=prediction_result['threat_reasons'] or []
        )
    else:
        print("Prediction failed.")