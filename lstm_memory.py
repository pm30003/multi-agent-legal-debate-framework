#!/usr/bin/env python3
"""
LSTM-Based Long Term Memory System for Legal Debate Framework
Learns patterns from previous debates and improves reasoning over time
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDebateDataset(Dataset):
    """PyTorch Dataset for legal debate sequences"""
    
    def __init__(self, sequences, labels, sequence_length=10):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LegalLSTMNetwork(nn.Module):
    """LSTM Network for learning legal debate patterns"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LegalLSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # [consensus_prediction, winning_side, confidence]
        )
        
        # Regression head for consensus prediction
        self.consensus_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output for classification
        last_output = attn_output[:, -1, :]
        
        # Generate predictions
        classification_output = self.classifier(last_output)
        consensus_prediction = self.consensus_head(last_output)
        
        return {
            'classification': classification_output,
            'consensus_prediction': consensus_prediction.squeeze(),
            'attention_weights': attn_weights,
            'hidden_states': lstm_out
        }

class LSTMLongTermMemory:
    """LSTM-based long-term memory system for legal debates"""
    
    def __init__(self, memory_dir: str = "lstm_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,  # Reduced for efficiency
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        
        # Database for storing debates
        self.db_path = self.memory_dir / "legal_debates.db"
        self.init_database()
        
        # Model parameters
        self.sequence_length = 10  # Number of rounds to consider
        self.feature_size = None
        
        # Load existing model if available
        self.load_model()
        
        logger.info(f"LSTM Memory system initialized on {self.device}")
    
    def init_database(self):
        """Initialize SQLite database for storing debate history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS debates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_description TEXT,
                legal_question TEXT,
                jurisdiction TEXT,
                rounds INTEGER,
                final_consensus REAL,
                consensus_reached BOOLEAN,
                plaintiff_arguments TEXT,
                defendant_arguments TEXT,
                judicial_decision TEXT,
                consensus_progression TEXT,
                timestamp DATETIME,
                case_category TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                debate_id INTEGER,
                round_number INTEGER,
                plaintiff_features TEXT,
                defendant_features TEXT,
                consensus_score REAL,
                convergence_score REAL,
                similarity_score REAL,
                FOREIGN KEY (debate_id) REFERENCES debates (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
    
    def extract_features(self, text: str) -> np.ndarray:
        """Extract features from legal text using TF-IDF and additional metrics"""
        
        # Basic TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()[0]
        
        # Additional legal-specific features
        legal_keywords = [
            'constitution', 'supreme court', 'high court', 'precedent', 'statute',
            'section', 'article', 'act', 'law', 'legal', 'court', 'judgment',
            'plaintiff', 'defendant', 'evidence', 'witness', 'contract',
            'breach', 'damages', 'remedy', 'injunction', 'liability'
        ]
        
        text_lower = text.lower()
        legal_feature_count = sum(1 for keyword in legal_keywords if keyword in text_lower)
        
        # Text statistics
        text_length = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = text_length / max(sentence_count, 1)
        
        # Combine all features
        additional_features = np.array([
            legal_feature_count,
            text_length,
            sentence_count,
            avg_sentence_length,
            text.count(','),  # Complexity indicator
            len(text.split())  # Word count
        ])
        
        return np.concatenate([tfidf_features, additional_features])
    
    def store_debate(self, debate_data: Dict[str, Any]) -> int:
        """Store a completed debate in the database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract main debate info
        multi_agent_debate = debate_data.get('multi_agent_debate', {})
        consensus_summary = multi_agent_debate.get('consensus_summary', {})
        
        # Determine case category based on keywords
        case_description = debate_data.get('case_description', '')
        case_category = self._categorize_case(case_description)
        
        # Insert main debate record
        cursor.execute('''
            INSERT INTO debates (
                case_description, legal_question, jurisdiction, rounds,
                final_consensus, consensus_reached, plaintiff_arguments,
                defendant_arguments, judicial_decision, consensus_progression,
                timestamp, case_category
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            debate_data.get('case_description', ''),
            debate_data.get('legal_question', ''),
            debate_data.get('jurisdiction', 'indian'),
            multi_agent_debate.get('debate_rounds', 0),
            consensus_summary.get('final_consensus_score', 0.0),
            consensus_summary.get('consensus_reached', False),
            json.dumps(multi_agent_debate.get('all_plaintiff_arguments', [])),
            json.dumps(multi_agent_debate.get('all_defendant_arguments', [])),
            multi_agent_debate.get('judicial_decision', ''),
            json.dumps(multi_agent_debate.get('consensus_progression', [])),
            datetime.now(),
            case_category
        ))
        
        debate_id = cursor.lastrowid
        
        # Store round-by-round features for training
        plaintiff_args = multi_agent_debate.get('all_plaintiff_arguments', [])
        defendant_args = multi_agent_debate.get('all_defendant_arguments', [])
        consensus_progression = multi_agent_debate.get('consensus_progression', [])
        
        for round_num, (p_arg, d_arg) in enumerate(zip(plaintiff_args, defendant_args)):
            if round_num < len(consensus_progression):
                metrics = consensus_progression[round_num]
                
                # Extract features (will be done later during training preparation)
                cursor.execute('''
                    INSERT INTO training_features (
                        debate_id, round_number, plaintiff_features, defendant_features,
                        consensus_score, convergence_score, similarity_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    debate_id, round_num + 1, p_arg, d_arg,
                    metrics.get('consensus_score', 0.0),
                    metrics.get('convergence', 0.0),
                    metrics.get('similarity', 0.0)
                ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored debate {debate_id} with {len(plaintiff_args)} rounds")
        return debate_id
    
    def _categorize_case(self, case_description: str) -> str:
        """Categorize legal case based on description"""
        
        case_lower = case_description.lower()
        
        if any(keyword in case_lower for keyword in ['salary', 'wage', 'payment', 'employment']):
            return 'employment'
        elif any(keyword in case_lower for keyword in ['contract', 'breach', 'agreement']):
            return 'contract'
        elif any(keyword in case_lower for keyword in ['property', 'real estate', 'land']):
            return 'property'
        elif any(keyword in case_lower for keyword in ['consumer', 'product', 'warranty']):
            return 'consumer'
        elif any(keyword in case_lower for keyword in ['accident', 'injury', 'negligence']):
            return 'tort'
        else:
            return 'general'
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from stored debates"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all training features
        cursor.execute('''
            SELECT tf.plaintiff_features, tf.defendant_features, tf.consensus_score,
                   tf.convergence_score, tf.similarity_score, d.case_category,
                   d.final_consensus, d.consensus_reached
            FROM training_features tf
            JOIN debates d ON tf.debate_id = d.id
            ORDER BY tf.debate_id, tf.round_number
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if len(results) < 10:
            logger.warning("Insufficient data for training. Need at least 10 samples.")
            return None, None
        
        # Extract text for TF-IDF fitting
        all_texts = []
        for row in results:
            all_texts.extend([row[0], row[1]])  # plaintiff and defendant features
        
        # Fit TF-IDF if not already fitted
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or self.tfidf_vectorizer.vocabulary_ is None:
            self.tfidf_vectorizer.fit(all_texts)
        
        # Prepare sequences
        sequences = []
        labels = []
        
        current_sequence = []
        current_labels = []
        
        for row in results:
            plaintiff_text, defendant_text = row[0], row[1]
            consensus_score, convergence_score, similarity_score = row[2], row[3], row[4]
            case_category = row[5]
            final_consensus, consensus_reached = row[6], row[7]
            
            # Extract features
            p_features = self.extract_features(plaintiff_text)
            d_features = self.extract_features(defendant_text)
            
            # Combine features
            round_features = np.concatenate([
                p_features, d_features,
                [consensus_score, convergence_score, similarity_score],
                [1 if case_category == cat else 0 for cat in ['employment', 'contract', 'property', 'consumer', 'tort', 'general']]
            ])
            
            current_sequence.append(round_features)
            current_labels.append([final_consensus, 1 if consensus_reached else 0, consensus_score])
            
            # Create sequence when we have enough data
            if len(current_sequence) >= self.sequence_length:
                sequences.append(current_sequence[-self.sequence_length:])
                labels.append(current_labels[-1])  # Use final label
                
                # Slide window
                current_sequence = current_sequence[1:]
                current_labels = current_labels[1:]
        
        if not sequences:
            logger.warning("No sequences could be created")
            return None, None
        
        # Convert to numpy arrays
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # Normalize features
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        sequences_flat = self.scaler.fit_transform(sequences_flat)
        sequences = sequences_flat.reshape(original_shape)
        
        # Store feature size
        self.feature_size = sequences.shape[-1]
        
        logger.info(f"Prepared {len(sequences)} sequences with {self.feature_size} features each")
        return sequences, labels
    
    def train_model(self, epochs: int = 50, batch_size: int = 16, learning_rate: float = 0.001):
        """Train the LSTM model on stored debate data"""
        
        # Prepare training data
        sequences, labels = self.prepare_training_data()
        
        if sequences is None or labels is None:
            logger.error("Cannot train model: insufficient data")
            return False
        
        # Create dataset and dataloader
        dataset = LegalDebateDataset(sequences, labels, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        if self.model is None:
            self.model = LegalLSTMNetwork(input_size=self.feature_size).to(self.device)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_sequences, batch_labels in dataloader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                
                # Calculate losses
                consensus_loss = mse_loss(outputs['consensus_prediction'], batch_labels[:, 0])
                classification_loss = bce_loss(outputs['classification'][:, 1], batch_labels[:, 1])
                
                # Combined loss
                total_batch_loss = consensus_loss + 0.5 * classification_loss
                
                # Backward pass
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model()
        
        logger.info(f"Training completed. Best loss: {best_loss:.4f}")
        return True
    
    def predict_debate_outcome(self, current_arguments: List[str], round_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Predict debate outcome based on current state"""
        
        if self.model is None or len(current_arguments) < 2:
            return {
                'predicted_consensus': 0.5,
                'confidence': 0.0,
                'winning_side': 'unknown',
                'should_continue': True
            }
        
        try:
            # Prepare current sequence
            sequence = []
            
            for i in range(0, len(current_arguments) - 1, 2):
                if i + 1 < len(current_arguments):
                    p_text = current_arguments[i]
                    d_text = current_arguments[i + 1]
                    
                    # Extract features
                    p_features = self.extract_features(p_text)
                    d_features = self.extract_features(d_text)
                    
                    # Get metrics for this round
                    round_idx = i // 2
                    if round_idx < len(round_metrics):
                        metrics = round_metrics[round_idx]
                        consensus_score = metrics.get('consensus_score', 0.0)
                        convergence_score = metrics.get('convergence', 0.0)
                        similarity_score = metrics.get('similarity', 0.0)
                    else:
                        consensus_score = convergence_score = similarity_score = 0.0
                    
                    # Combine features (simplified case category)
                    round_features = np.concatenate([
                        p_features, d_features,
                        [consensus_score, convergence_score, similarity_score],
                        [0, 0, 0, 0, 0, 1]  # Default to 'general' category
                    ])
                    
                    sequence.append(round_features)
            
            if len(sequence) < self.sequence_length:
                # Pad sequence if too short
                while len(sequence) < self.sequence_length:
                    sequence.insert(0, np.zeros_like(sequence[0]) if sequence else np.zeros(self.feature_size))
            else:
                # Take last sequence_length items
                sequence = sequence[-self.sequence_length:]
            
            # Convert to tensor
            sequence_array = np.array([sequence])
            sequence_array = self.scaler.transform(sequence_array.reshape(-1, sequence_array.shape[-1])).reshape(sequence_array.shape)
            sequence_tensor = torch.FloatTensor(sequence_array).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                
                predicted_consensus = outputs['consensus_prediction'].cpu().numpy()[0]
                classification_output = torch.softmax(outputs['classification'], dim=1).cpu().numpy()[0]
                
                return {
                    'predicted_consensus': float(predicted_consensus),
                    'confidence': float(classification_output[2]),  # confidence score
                    'winning_side': 'plaintiff' if classification_output[0] > 0.5 else 'defendant',
                    'should_continue': predicted_consensus < 0.7,
                    'attention_weights': outputs['attention_weights'].cpu().numpy()[0].tolist()
                }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'predicted_consensus': 0.5,
                'confidence': 0.0,
                'winning_side': 'unknown',
                'should_continue': True
            }
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from the long-term memory system"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('SELECT COUNT(*) FROM debates')
        total_debates = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(final_consensus), AVG(rounds) FROM debates')
        avg_consensus, avg_rounds = cursor.fetchone()
        
        cursor.execute('SELECT COUNT(*) FROM debates WHERE consensus_reached = 1')
        successful_debates = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT case_category, COUNT(*), AVG(final_consensus), AVG(rounds)
            FROM debates
            GROUP BY case_category
        ''')
        category_stats = cursor.fetchall()
        
        conn.close()
        
        success_rate = (successful_debates / total_debates * 100) if total_debates > 0 else 0
        
        return {
            'total_debates': total_debates,
            'average_consensus': float(avg_consensus) if avg_consensus else 0.0,
            'average_rounds': float(avg_rounds) if avg_rounds else 0.0,
            'success_rate': success_rate,
            'model_trained': self.model is not None,
            'feature_size': self.feature_size,
            'category_statistics': {
                cat: {
                    'count': count,
                    'avg_consensus': float(avg_cons),
                    'avg_rounds': float(avg_rds)
                }
                for cat, count, avg_cons, avg_rds in category_stats
            }
        }
    
    def save_model(self):
        """Save the trained model and preprocessing components"""
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'feature_size': self.feature_size,
                'sequence_length': self.sequence_length,
            }, self.memory_dir / 'lstm_model.pth')
        
        # Save preprocessing components
        with open(self.memory_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(self.memory_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info("Model and preprocessors saved successfully")
    
    def load_model(self):
        """Load trained model and preprocessing components"""
        
        model_path = self.memory_dir / 'lstm_model.pth'
        tfidf_path = self.memory_dir / 'tfidf_vectorizer.pkl'
        scaler_path = self.memory_dir / 'scaler.pkl'
        
        try:
            if model_path.exists() and tfidf_path.exists() and scaler_path.exists():
                # Load model
                checkpoint = torch.load(model_path, map_location=self.device)
                self.feature_size = checkpoint['feature_size']
                self.sequence_length = checkpoint['sequence_length']
                
                self.model = LegalLSTMNetwork(input_size=self.feature_size).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Load preprocessors
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info("Model and preprocessors loaded successfully")
            else:
                logger.info("No existing model found. Will create new one after training.")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None

# Usage example and testing
if __name__ == "__main__":
    
    # Initialize LSTM memory system
    lstm_memory = LSTMLongTermMemory()
    
    # Example: Store a debate (you would get this from your main system)
    sample_debate = {
        'case_description': 'Company withheld employee salaries for 3 months',
        'legal_question': 'What remedies under Payment of Wages Act?',
        'jurisdiction': 'indian',
        'multi_agent_debate': {
            'debate_rounds': 3,
            'all_plaintiff_arguments': [
                'The company violated the Payment of Wages Act by withholding salaries.',
                'Section 3 clearly mandates timely payment of wages to employees.',
                'The Supreme Court precedents support immediate payment relief.'
            ],
            'all_defendant_arguments': [
                'The company faced legitimate financial difficulties beyond control.',
                'Force majeure conditions justified the temporary delay in payments.',
                'The company has now arranged for immediate salary disbursement.'
            ],
            'consensus_progression': [
                {'consensus_score': 0.2, 'convergence': 0.1, 'similarity': 0.3},
                {'consensus_score': 0.4, 'convergence': 0.3, 'similarity': 0.5},
                {'consensus_score': 0.6, 'convergence': 0.5, 'similarity': 0.7}
            ],
            'consensus_summary': {
                'final_consensus_score': 0.6,
                'consensus_reached': False
            },
            'judicial_decision': 'The court finds in favor of the employees...'
        }
    }
    
    # Store the debate
    debate_id = lstm_memory.store_debate(sample_debate)
    print(f"Stored debate with ID: {debate_id}")
    
    # Get insights
    insights = lstm_memory.get_memory_insights()
    print("Memory insights:", insights)
    
    # Train model (if enough data)
    if insights['total_debates'] >= 5:
        print("Training LSTM model...")
        lstm_memory.train_model(epochs=20)
        
        # Test prediction
        current_args = sample_debate['multi_agent_debate']['all_plaintiff_arguments'] + sample_debate['multi_agent_debate']['all_defendant_arguments']
        metrics = sample_debate['multi_agent_debate']['consensus_progression']
        
        prediction = lstm_memory.predict_debate_outcome(current_args, metrics)
        print("Prediction:", prediction)
    else:
        print("Need more debates to train the model")