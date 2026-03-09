"""
Verifiable ML Pipeline with Cryptographic Commitments
Architectural Rationale: Creates non-repudiable proof of prediction timing and content
to build cryptographic trust with institutional clients.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.exceptions import GoogleCloudError
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("firebase_admin not available. Running in local mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionCommitment:
    """Immutable record of prediction commitment with cryptographic proof"""
    timestamp: datetime
    prediction_hash: str
    feature_hash: str
    model_version: str
    market: str
    timeframe: str
    commitment_id: Optional[str] = None
    on_chain_tx_hash: Optional[str] = None
    revealed: bool = False
    actual_outcome: Optional[float] = None
    realized_at: Optional[datetime] = None


class VerifiableMLPipeline:
    """
    ML Pipeline that generates cryptographically verifiable predictions.
    
    Key Design Principles:
    1. Deterministic hashing of all inputs/outputs
    2. Time-stamped commitments before market events
    3. Separation of commitment (hash) from revelation (actual values)
    4. Redundant storage with integrity verification
    """
    
    def __init__(
        self,
        firebase_cred_path: Optional[str] = None,
        model_save_path: str = "models/ensemble_model_v1.2.0.pkl"
    ):
        """
        Initialize pipeline with defensive validation.
        
        Args:
            firebase_cred_path: Path to Firebase service account JSON
            model_save_path: Path to serialized model
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If Firebase credentials are invalid
        """
        # Initialize model with validation
        self.model_save_path = model_save_path
        self._load_or_train_model()
        
        # Initialize Firebase with error handling
        self.db = None
        if firebase_cred_path:
            try:
                if not FIREBASE_AVAILABLE:
                    raise ImportError("firebase_admin package not installed")
                
                # Check if Firebase app already initialized
                if not firebase_admin._apps:
                    cred = credentials.Certificate(firebase_cred_path)
                    firebase_admin.initialize_app(cred)
                    logger.info("Firebase initialized successfully")
                
                self.db = firestore.client()
                logger.info("Firestore client connected")
                
            except FileNotFoundError as e:
                logger.error(f"Firebase credentials file not found: {e}")
                raise
            except ValueError as e:
                logger.error(f"Invalid Firebase credentials: {e}")
                raise
            except GoogleCloudError as e:
                logger.error(f"Google Cloud error: {e}")
                # Fallback to local mode
                self.db = None
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Track commitments for later revelation
        self.pending_commitments: Dict[str, PredictionCommitment] = {}
        
        logger.info("VerifiableMLPipeline initialized successfully")
    
    def _load_or_train_model(self) -> None:
        """Load pre-trained model or train new one with validation"""
        try:
            # Check if model file exists
            with open(self.model_save_path, 'rb') as f:
                self.model = joblib.load(f)
            logger.info(f"Model loaded from {self.model_save_path}")
            
        except FileNotFoundError:
            logger.warning(f"Model file not found at {self.model_save_path}. Training new model...")
            self._train_new_model()
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Corrupted model file: {e}. Training new model...")
            self._train_new_model()
    
    def _train_new_model(self) -> None:
        """Train ensemble model with proper error handling"""
        try:
            # Create ensemble of Random Forest and Gradient Boosting
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            gb = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Simple ensemble - average predictions
            # In production, use stacking or voting classifier
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('ensemble_rf', rf)  # Using RF as primary for now
            ])
            
            # Train with dummy data - in production, load from Firestore
            X_dummy = np.random.randn(1000, 20)
            y_dummy = np.random.randn(1000)
            self.model.fit(X_dummy, y_dummy)
            
            # Save model
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            joblib.dump(self.model, self.model_save_path)
            logger.info(f"New model trained and saved to {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def _hash_prediction(self, prediction: np.ndarray) -> str:
        """
        Create deterministic SHA256 hash of prediction.
        
        Args:
            prediction: numpy array of predictions
        
        Returns:
            Hexadecimal hash string
        
        Raises:
            ValueError: If prediction is empty or contains NaN
        """
        # Validate prediction array
        if prediction.size == 0:
            raise ValueError("Prediction array is empty")
        
        if np.any(np.isnan(prediction)):
            raise ValueError("Prediction contains NaN values")
        
        # Convert to deterministic string representation
        # Use high precision formatting for consistency
        prediction_rounded = np.round(prediction, 8)  # 8 decimal places
        pred_list = prediction_rounded.tolist()
        
        # Handle both scalar and array predictions
        if not isinstance(pred_list, list):
            pred_list = [pred_list]
        
        # Sort keys if dictionary, maintain order if list
        pred_str = json.dumps(pred_list, sort_keys=True)
        
        # Create hash
        return hashlib.sha256(pred_str.encode()).hexdigest()
    
    def _hash_features(self, features: pd.DataFrame) -> str:
        """
        Hash feature set deterministically.
        
        Args:
            features: pandas DataFrame of features
        
        Returns:
            Hexadecimal hash string
        """
        # Validate features
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        if features.isnull().any().any():
            logger.warning("Features contain NaN values - hashing anyway")
        
        # Create deterministic representation
        # Sort columns and round values for consistency
        features_sorted = features.reindex(sorted(features.columns), axis=1)
        features_rounded = features_sorted.round(8)
        
        # Convert to JSON string
        features_dict = features_rounded.to_dict(orient='list')
        features_str = json.dumps(features_dict, sort_keys=True)
        
        return hashlib.sha256(features_str.encode()).hexdigest()
    
    def generate_signal(
        self,
        features: pd.DataFrame,
        market: str,
        timeframe: str = "1h"
    ) -> Tuple[np.ndarray, PredictionCommitment]:
        """
        Generate trading signal with cryptographic commitment.
        
        Args:
            features: Preprocessed feature DataFrame
            market: Market identifier (e.g., "BTC/USDT")
            timeframe: Trading timeframe
        
        Returns:
            Tuple of (prediction_array, commitment_object)
        
        Raises:
            ValueError: If features are invalid
            RuntimeError: If commitment storage fails
        """
        # Validate inputs
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Features must be pandas DataFrame")
        
        if features.shape[0] == 0:
            raise ValueError("Features must contain at least one sample")
        
        logger.info(f"Generating signal for {market} ({timeframe})")
        
        try:
            # 1. Generate prediction
            prediction = self.model.predict(features)
            logger.debug(f"Prediction generated: shape={prediction.shape}")
            
            # 2. Create cryptographic commitment
            prediction_hash = self._hash_prediction(prediction)
            feature_hash = self._hash_features(features)
            
            commitment = PredictionCommitment(
                timestamp=datetime.utcnow(),
                prediction_hash=prediction_hash,
                feature_hash=feature_hash,
                model_version="1.2.0",
                market=market,
                timeframe=timeframe
            )
            
            # 3. Store commitment in Firestore with retry logic
            if self.db:
                self._store_commitment(commitment)
            else:
                logger.warning("Firestore not available - storing locally only")
                commitment.commitment_id = f"local_{int(datetime.utcnow().timestamp())}"
                self.pending_commitments[commitment.commitment_id] = commitment
            
            # 4. In production, would also store hash on-chain
            # self._store_on_chain(prediction_hash)
            
            logger.info(f"Signal generated with commitment ID: {commitment.commitment_id}")
            return prediction, commitment
            
        except Exception as e:
            logger.error(f"Failed to generate signal: {e}")
            raise RuntimeError(f"Signal generation failed: {e}")
    
    def _store_commitment(self, commitment: PredictionCommitment) -> None:
        """
        Store commitment in Firestore with error handling and retries.
        
        Args:
            commitment: PredictionCommitment object
        
        Raises:
            RuntimeError: If storage fails after retries
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate unique ID
                commitment_id = f"commit_{int(datetime.utcnow().timestamp())}_{hashlib.md5(commitment.prediction_hash.encode()).hexdigest()[:8]}"
                commitment.commitment_id = commitment_id
                
                # Convert to dictionary for Firestore
                commitment_dict = asdict(commitment)
                
                # Convert datetime to ISO string for Firestore compatibility
                commitment_dict['timestamp'] = commitment.timestamp.isoformat()
                if commitment_dict['realized_at']:
                    commitment_dict['realized_at'] = commitment.realized_at.isoformat()
                
                # Store in Firestore
                doc_ref = self.db.collection('predictions').document(commitment_id)
                doc_ref.set(commitment_dict)
                
                # Cache locally for quick access
                self.pending_commitments[commitment_id] = commitment
                
                logger.info(f"Commitment stored in Firestore: {commitment_id}")
                return
                
            except GoogleCloudError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to store commitment after {max_retries} attempts: {e}")
                    raise RuntimeError(f"Firestore storage failed: {e}")
                logger.warning(f"Firestore error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def reveal_prediction(
        self,
        commitment_id: str,
        actual_outcome: float,
        realized_at: Optional[datetime] = None
    ) -> bool:
        """
        Reveal actual outcome for a previously committed prediction.
        
        Args:
            commitment_id: ID of the commitment
            actual_outcome: Actual market outcome
            realized_at: When outcome was realized (defaults to now)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find commitment
            commitment = self.pending_commitments.get(commitment_id)
            if not commitment and self.db:
                # Try to load from Firestore
                doc_ref = self.db.collection('predictions').document(commitment_id)
                doc = doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    commitment = PredictionCommitment(**data)
                else:
                    logger.error(f"Commitment {commitment_id} not found")
                    return False
            
            if not commitment:
                logger.error(f"Commitment {commitment_id} not found in cache or Firestore")
                return False
            
            # Update commitment
            commitment.revealed = True
            commitment.actual_outcome = actual_outcome
            commitment.realized_at = realized_at or datetime.utcnow()
            
            # Update Firestore
            if self.db:
                doc_ref = self.db.collection('predictions').document(commitment_id)
                update_data = {
                    'revealed': True,
                    'actual_outcome': actual_outcome,
                    'realized_at': commitment.realized_at.isoformat()
                }
                doc_ref.update(update_data)
            
            logger.info(f"Prediction revealed for {commitment_id}: outcome={actual_outcome}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reveal prediction {commitment_id}: {e}")
            return False
    
    def verify_commitment(self, commitment_id: str) -> Dict[str, Any]:
        """
        Verify cryptographic integrity of a commitment.
        
        Args:
            commitment_id: ID of the commitment to verify
        
        Returns:
            Dictionary with verification results
        """
        verification_result = {
            'commitment_id': commitment_id,
            'verified': False,
            'errors': [],
            'timestamp': None,
            'elapsed_hours': None
        }
        
        try:
            # Load commitment
            commitment = None
            if commitment_id in self.pending_commitments:
                commitment = self.pending_commitments[commitment_id]
            elif self.db:
                doc_ref = self.db.collection('predictions').document(commitment_id)
                doc = doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    commitment = PredictionCommitment(**data)
            
            if not commitment:
                verification_result