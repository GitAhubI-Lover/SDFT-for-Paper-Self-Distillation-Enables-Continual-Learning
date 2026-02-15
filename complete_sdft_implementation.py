"""
Complete PyTorch Implementation of Self-Distillation Fine-Tuning (SDFT)
based on the paper "Self-Distillation Enables Continual Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import copy
import logging
from typing import List, Tuple, Optional, Dict, Any


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SDFTDataset(Dataset):
    """
    Dataset class for SDFT training data containing (query, demonstration) pairs
    """
    def __init__(self, data: List[Tuple[str, str]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        query, demonstration = self.data[idx]
        return {
            'query': query,
            'demonstration': demonstration
        }


class SDFTEncoder(nn.Module):
    """
    Encoder wrapper for student and teacher models in SDFT
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.model.config
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs


class SDFTModel(nn.Module):
    """
    Complete SDFT model with student and teacher components
    """
    def __init__(self, model_name: str, ema_alpha: float = 0.02):
        super().__init__()
        
        # Student model that gets updated during training
        self.student_encoder = SDFTEncoder(model_name)
        
        # Teacher model that is updated via EMA of student parameters
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        
        # Freeze teacher initially
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
            
        self.ema_alpha = ema_alpha
    
    def create_teacher_context(self, query: str, demonstration: str) -> str:
        """Create teacher context following paper's prompt template"""
        return (
            f"<Question>\n{query}\n"
            f"This is an example for a response to the question:\n"
            f"<Demonstration>\n{demonstration}\n"
            f"Now answer with a response of your own, including the thinking process:"
        )
    
    def create_student_context(self, query: str) -> str:
        """Create student context (query only)"""
        return query
    
    def update_teacher_via_ema(self):
        """Update teacher parameters using exponential moving average of student parameters"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_encoder.parameters(),
                self.student_encoder.parameters()
            ):
                teacher_param.data.mul_(1.0 - self.ema_alpha).add_(
                    student_param.data, alpha=self.ema_alpha
                )
    
    def get_student_output(self, input_ids, attention_mask=None):
        """Get output from student model"""
        return self.student_encoder(input_ids, attention_mask)
    
    def get_teacher_output(self, input_ids, attention_mask=None):
        """Get output from teacher model"""
        return self.teacher_encoder(input_ids, attention_mask)


class SDFTLoss(nn.Module):
    """
    Loss function implementing reverse KL divergence for SDFT
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute reverse KL divergence: D_KL(π_θ(·|x) || π(·|x, c))
        Where student is π_θ(·|x) and teacher is π(·|x, c)
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
        """
        # Apply softmax to teacher logits to get probability distribution
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Apply log_softmax to student logits to get log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        # Compute reverse KL divergence: sum(teacher_probs * (log(teacher_probs) - student_log_probs))
        # Though we don't need the log(teacher_probs) term for gradients, 
        # for the true reverse KL we compute: sum(student_probs * log(student_probs / teacher_probs))
        # So we just use: sum(teacher_probs * (log(teacher_probs + eps) - student_log_probs)) 
        # where the teacher_probs term cancels out in gradients, so we can simplify to:
        # -sum(teacher_probs * log_softmax(student_logits)) + const
        # But for true reverse KL: D(P||Q) = sum(P * log(P/Q)) = sum(P*log(P) - P*log(Q))
        # So: -sum(P*log(Q)) + const where P=teacher_probs and Q=student_probs
        # This is: -sum(teacher_probs * log_softmax(student_logits)) + entropy_term
        # But we implement the proper reverse KL: student_probs * log(student_probs / teacher_probs)
        
        student_probs = F.softmax(student_logits, dim=-1)
        kl_divergence = student_probs * (student_log_probs - torch.log(teacher_probs + 1e-12))
        return kl_divergence.sum(dim=-1).mean()


class SDFTOptimizer:
    """
    Optimizer wrapper for SDFT training
    """
    def __init__(self, model: SDFTModel, learning_rate: float = 1e-5):
        self.model = model
        self.optimizer = AdamW(
            [{'params': model.student_encoder.parameters()}],
            lr=learning_rate,
            weight_decay=0.0  # As per paper's hyperparameters
        )
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
        # Update teacher after optimizer step
        self.model.update_teacher_via_ema()


class SDFTTrainer:
    """
    Main trainer class for Self-Distillation Fine-Tuning
    """
    def __init__(
        self,
        model_name: str,
        train_data: List[Tuple[str, str]],
        val_data: Optional[List[Tuple[str, str]]] = None,
        learning_rate: float = 1e-5,
        ema_alpha: float = 0.02,
        batch_size: int = 4,
        max_length: int = 1024,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize model, tokenizer, and optimizer
        self.model = SDFTModel(model_name, ema_alpha).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.optimizer = SDFTOptimizer(self.model, learning_rate)
        self.loss_fn = SDFTLoss()
        
        # Create datasets
        self.train_dataset = SDFTDataset(train_data, self.tokenizer, max_length)
        self.val_dataset = SDFTDataset(val_data, self.tokenizer, max_length) if val_data else None
        
        # Data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False) if self.val_dataset else None
    
    def tokenize_batch(self, batch: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
        """
        Tokenize student and teacher inputs for a batch
        """
        queries = batch['query']
        demonstrations = batch['demonstration']
        
        # Student inputs (query only)
        student_texts = [
            self.model.create_student_context(query) for query in queries
        ]
        student_encodings = self.tokenizer(
            student_texts,
            truncation=True, 
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Teacher inputs (query + demonstration)
        teacher_texts = [
            self.model.create_teacher_context(query, demo) 
            for query, demo in zip(queries, demonstrations)
        ]
        teacher_encodings = self.tokenizer(
            teacher_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return (
            {k: v.to(self.device) for k, v in student_encodings.items()},
            {k: v.to(self.device) for k, v in teacher_encodings.items()}
        )
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Tokenize inputs
            student_inputs, teacher_inputs = self.tokenize_batch(batch)
            
            # Forward pass - get teacher outputs first (frozen)
            with torch.no_grad():
                teacher_outputs = self.model.get_teacher_output(
                    input_ids=teacher_inputs['input_ids'],
                    attention_mask=teacher_inputs['attention_mask']
                )
                teacher_logits = teacher_outputs.logits
            
            # Forward pass - get student outputs
            student_outputs = self.model.get_student_output(
                input_ids=student_inputs['input_ids'],
                attention_mask=student_inputs['attention_mask']
            )
            student_logits = student_outputs.logits
            
            # Ensure shapes match
            min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_seq_len, :]
            teacher_logits = teacher_logits[:, :min_seq_len, :]
            
            # Compute loss
            loss = self.loss_fn(student_logits, teacher_logits)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Processed {num_batches} batches, current loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model
        """
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Tokenize inputs
                student_inputs, teacher_inputs = self.tokenize_batch(batch)
                
                # Get outputs
                teacher_outputs = self.model.get_teacher_output(
                    input_ids=teacher_inputs['input_ids'],
                    attention_mask=teacher_inputs['attention_mask']
                )
                teacher_logits = teacher_outputs.logits
                
                student_outputs = self.model.get_student_output(
                    input_ids=student_inputs['input_ids'],
                    attention_mask=student_inputs['attention_mask']
                )
                student_logits = student_outputs.logits
                
                # Ensure shapes match
                min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
                student_logits = student_logits[:, :min_seq_len, :]
                teacher_logits = teacher_logits[:, :min_seq_len, :]
                
                # Compute loss
                loss = self.loss_fn(student_logits, teacher_logits)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(
        self, 
        num_epochs: int = 3,
        eval_every_n_epochs: int = 1,
        save_checkpoint: bool = False,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Main training loop
        """
        logger.info("Starting SDFT training...")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate if validation data is available
            val_loss = self.validate() if self.val_loader else 0.0
            
            logger.info(f"Epoch {epoch + 1} completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint if requested
            if save_checkpoint and ((epoch + 1) % eval_every_n_epochs == 0):
                checkpoint_path = f"{checkpoint_dir}/sdft_epoch_{epoch + 1}"
                self.save_model(checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        logger.info("Training completed!")
    
    def generate(self, query: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        Generate response for a query using the trained student model
        """
        self.model.eval()
        
        context = self.model.create_student_context(query)
        inputs = self.tokenizer(context, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.student_encoder.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part (after the context)
        response_start = len(context)
        response = generated_text[response_start:].strip()
        return response
    
    def save_model(self, path: str):
        """
        Save the trained student model
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save student model and tokenizer
        self.model.student_encoder.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def create_sample_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create sample training and validation data
    """
    # Training examples
    train_data = [
        (
            "What is the capital of France?",
            "The capital of France is Paris. Paris is located in northern France along the Seine River."
        ),
        (
            "Explain quantum computing in simple terms.",
            "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, allowing for parallel processing that can solve certain problems much faster than classical computers."
        ),
        (
            "How do neural networks learn?",
            "Neural networks learn through a process called backpropagation, where the network adjusts its weights based on the difference between predicted and actual outputs, gradually improving its accuracy over time."
        ),
        (
            "What are the benefits of renewable energy?",
            "Renewable energy sources like solar and wind power are sustainable, reduce greenhouse gas emissions, decrease dependence on fossil fuels, and can provide long-term economic benefits."
        ),
        (
            "Explain the water cycle.",
            "The water cycle consists of evaporation, condensation, and precipitation. Water evaporates from surfaces, condenses into clouds, and returns to Earth as precipitation."
        ),
        (
            "How does photosynthesis work?",
            "Photosynthesis occurs in plant chloroplasts where chlorophyll captures sunlight, converting carbon dioxide and water into glucose and oxygen using light energy."
        ),
        (
            "What's the difference between DNA and RNA?",
            "DNA is double-stranded and contains the genetic code with thymine, while RNA is single-stranded with uracil instead of thymine, and is involved in protein synthesis."
        ),
        (
            "Explain the theory of relativity.",
            "Einstein's theory of relativity describes how space and time are interwoven into spacetime, with gravity being the curvature of spacetime caused by mass and energy."
        )
    ]
    
    # Validation examples
    val_data = [
        (
            "What is the largest planet in our solar system?",
            "Jupiter is the largest planet in our solar system, with a diameter approximately 11 times that of Earth."
        ),
        (
            "How do vaccines work?",
            "Vaccines work by introducing a weakened, dead, or small piece of a pathogen to trigger an immune response, creating antibodies and memory cells for future protection."
        )
    ]
    
    return train_data, val_data


def main():
    """
    Main function to demonstrate SDFT implementation
    """
    # Create sample training and validation data
    train_data, val_data = create_sample_data()
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print("\nSample training example:")
    print(f"Query: {train_data[0][0]}")
    print(f"Demonstration: {train_data[0][1]}")
    
    # Initialize trainer with a smaller model for quick demonstration
    # For actual use, replace with a larger model like 'Qwen/Qwen2.5-7B-Instruct'
    trainer = SDFTTrainer(
        model_name="gpt2",  # Use gpt2 for demo; could use larger models in practice
        train_data=train_data,
        val_data=val_data,
        learning_rate=5e-6,  # Lower learning rate as per paper's hyperparameter table
        ema_alpha=0.02,      # As per paper's hyperparameter table
        batch_size=2,        # Adjust based on available GPU memory
        max_length=512       # Adjust based on sequence requirements
    )
    
    # Start training
    trainer.train(
        num_epochs=2,           # Few epochs for demo; paper used 2-4 epochs for skill learning
        eval_every_n_epochs=1,
        save_checkpoint=False   # Set to True to save checkpoints
    )
    
    # Test generation after training
    test_query = "What is machine learning?"
    response = trainer.generate(test_query)
    print(f"\nTest query: {test_query}")
    print(f"Generated response: {response}")


if __name__ == "__main__":
    main()