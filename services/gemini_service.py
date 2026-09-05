import os
import json
from dotenv import load_dotenv, set_key

load_dotenv()

USER_ACTIVE_API_KEY = os.getenv("GEMINI_API_KEY", "")

def set_runtime_api_key(key: str, persist: bool = True):
    """
    Sets the active Gemini API key in runtime and optionally persists to .env
    """
    global USER_ACTIVE_API_KEY
    USER_ACTIVE_API_KEY = key.strip()
    os.environ["GEMINI_API_KEY"] = USER_ACTIVE_API_KEY
    if persist:
        try:
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            if not os.path.exists(env_path):
                with open(env_path, 'w') as f:
                    f.write(f"GEMINI_API_KEY={USER_ACTIVE_API_KEY}\n")
            else:
                set_key(env_path, "GEMINI_API_KEY", USER_ACTIVE_API_KEY)
        except Exception as e:
            print(f"Warning: Could not persist GEMINI_API_KEY to .env: {e}")
    return True

def get_active_api_key():
    global USER_ACTIVE_API_KEY
    return USER_ACTIVE_API_KEY or os.getenv("GEMINI_API_KEY", "")

def get_gemini_client(api_key=None):
    """
    Initializes the Google GenAI SDK client if a key is provided or stored.
    """
    key = api_key or get_active_api_key()
    if not key or key.strip() == "" or "your_gemini_api_key" in key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key.strip())
        return client
    except Exception as e:
        print(f"Warning: Failed to initialize Google GenAI Client: {e}")
        return None

def validate_api_key(api_key: str):
    """
    Validates a Google Studio API Key by sending a test prompt to Gemini 2.5 Flash / 1.5 Flash.
    """
    if not api_key or not api_key.strip():
        return {'valid': False, 'error': 'API key cannot be empty'}
    try:
        from google import genai
        client = genai.Client(api_key=api_key.strip())
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # Test quick live generation
        response = client.models.generate_content(
            model=model_name,
            contents="Respond with 'API_KEY_VALID' if you receive this."
        )
        return {
            'valid': True,
            'model': model_name,
            'message': 'API Key successfully verified with Google AI Studio'
        }
    except Exception as e:
        err_msg = str(e)
        return {
            'valid': False,
            'error': err_msg
        }

def generate_dataset_explanation(structured_json_summary, api_key=None):
    """
    Calls Google Gemini API with structured model prediction JSON to produce:
    - Executive Research Readiness Narrative
    - Methodological Risks & Downstream ML Threats
    - Step-by-Step Actionable Data Cleaning Plan
    """
    client = get_gemini_client(api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    prompt = f"""
You are the Lead Neuroimaging ML Quality Control Scientist at NeuroData Quality AI.
Your role is to explain structured neuroimaging dataset quality and research-readiness predictions to researchers.

CRITICAL INSTRUCTIONS:
1. You do NOT make the core quality predictions. The PyTorch and Scikit-learn models have already computed all metrics and readiness scores.
2. Read the structured JSON predictions below and provide an expert, publication-grade, concise synthesis.
3. Your explanation MUST be strictly grounded in the provided JSON metrics.

STRUCTURED DATASET QC JSON:
{json.dumps(structured_json_summary, indent=2)}

Please generate a JSON response with exactly the following keys:
{{
  "executive_summary": "A concise 2-3 sentence overview of the cohort quality, readiness status, and core findings.",
  "risk_analysis": "Bullet points detailing potential machine learning risks (e.g. gradient corruption from motion artifacts, site-confound bias, low SNR, FOV truncations).",
  "recommended_actions": [
    "Actionable step 1 for data cleaning / exclusion / harmonization",
    "Actionable step 2",
    "Actionable step 3"
  ],
  "technical_definitions": "Brief explanation of key metrics (SNR, CNR, Blur variance, Isolation Forest anomaly) for the researcher."
}}

Output valid JSON only. Do not enclose in markdown code blocks if possible, or use standard ```json ... ``` blocks.
"""
    if client:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            text = response.text.strip()
            # Clean markdown codeblocks if returned
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())
            return parsed
        except Exception as e:
            print(f"Gemini API invocation error: {e}. Falling back to deterministic scientific engine.")
            
    # Deterministic scientific synthesis fallback
    readiness_label = structured_json_summary.get('readiness_label', 'Requires Cleaning')
    score = structured_json_summary.get('readiness_score', 65.0)
    total = structured_json_summary.get('total_scans', 0)
    good = structured_json_summary.get('good_count', 0)
    poor = structured_json_summary.get('poor_count', 0)
    unusable = structured_json_summary.get('unusable_count', 0)
    outliers = structured_json_summary.get('outlier_count', 0)
    
    summary = f"The dataset achieved an overall Research Readiness Score of {score}/100 and is categorized as '{readiness_label}'. Out of {total} evaluated MRI scans, {good} meet pristine research standards, while {poor} exhibit motion/noise degradation and {unusable} require strict exclusion."
    
    risks = (
        f"• Downstream ML Generalization: Including {unusable} unusable scans will introduce spurious high-loss gradients during deep network training.\n"
        f"• Anomaly Contamination: Isolation Forest detected {outliers} outlier scan(s) with aberrant spatial or frequency profiles.\n"
        f"• Scanner Variance: Multi-site hardware differences pose a risk of batch confounding if site harmonization is not applied."
    )
    
    actions = [
        f"Prune {unusable} unusable scan(s) from the training and validation splits immediately.",
        f"Apply spatial filtering or N4ITK bias field correction on scans with elevated intensity gradients.",
        "Perform ComBat or z-score feature harmonization across distinct scanner protocols before model fitting."
    ]
    
    defs = "SNR (Signal-to-Noise Ratio) measures signal integrity over background noise. Laplacian variance quantifies edge sharpness. Isolation Forest detects scans outside the normative cohort distribution."
    
    return {
        'executive_summary': summary,
        'risk_analysis': risks,
        'recommended_actions': actions,
        'technical_definitions': defs
    }

def answer_researcher_question(question, dataset_context, api_key=None):
    """
    Interactive real-time Q&A engine powered by Gemini to answer any researcher question about their dataset or neuroimaging AI in real-time.
    """
    client = get_gemini_client(api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    prompt = f"""
You are the NeuroData Quality AI Research Assistant, an expert neuroimaging computer scientist, data quality specialist, and deep learning researcher.
You are helping a researcher with quality control, artifact correction, MRI physics, and deep learning readiness.

ACTIVE DATASET CONTEXT:
{json.dumps(dataset_context, indent=2) if dataset_context else 'No specific dataset loaded. General neuroimaging QC & AI research consultation.'}

USER QUESTION:
{question}

Provide an accurate, insightful, clear, and comprehensive answer formatted with markdown headings and bullet points where helpful. Incorporate relevant neuroimaging details (e.g. k-space, phase-encoding ghosting, B1 field inhomogeneity, ComBat harmonization, PyTorch CNN transforms, Grad-CAM attention).
"""
    if client:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini Live Call Error: {e}")
            return f"**Gemini API Notice**: Received error `{str(e)}`. Please verify your Google AI Studio API key in settings or try again."
            
    # Fallback response if no API key is provided
    q_lower = question.lower()
    if "api key" in q_lower or "google studio" in q_lower or "setup" in q_lower:
        return "To activate live real-time Gemini responses, enter your Google AI Studio API key in the **Google Studio API Key** settings box in the top right corner. Once saved, Gemini 2.5 Flash will answer all questions live!"
    elif "unusable" in q_lower or "exclude" in q_lower or "poor" in q_lower:
        return f"Based on our PyTorch quality model, scans flagged as 'Unusable' (such as those with severe FOV truncation or missing slice data) should be excluded prior to dataset splitting. For 'Poor' scans exhibiting motion ghosting, you can either apply retrospective motion correction or remove them to ensure the model doesn't overfit on artifact patterns.\n\n*(Note: Add your Google Studio API key for live custom AI answers)*"
    elif "grad-cam" in q_lower or "explain" in q_lower or "heatmap" in q_lower:
        return f"Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradients of the predicted quality class score with respect to the feature maps of the final convolutional layer. The glowing red/orange regions highlight where motion phase errors or anatomical dropouts triggered quality penalties.\n\n*(Note: Add your Google Studio API key for live custom AI answers)*"
    elif "readiness" in q_lower or "score" in q_lower:
        score = dataset_context.get('readiness_score', 65.0) if dataset_context else 65.0
        label = dataset_context.get('readiness_label', 'Requires Cleaning') if dataset_context else 'Requires Cleaning'
        return f"The current dataset has a Research-Readiness score of {score}/100 ({label}). This composite score combines image quality classification (35%), SNR & motion suppression (20%), metadata completeness (20%), outlier isolation (15%), and scanner parity (10%).\n\n*(Note: Add your Google Studio API key for live custom AI answers)*"
    else:
        return f"NeuroData Quality AI has evaluated your request. For full real-time answers to arbitrary questions powered by Google Gemini, please enter your Google AI Studio API Key in the top navigation bar **'🔑 Google Studio API Key'** button."
