"""
Implementation of Self-Distillation Fine-Tuning (SDFT) 
based on the paper \"Self-Distillation Enables Continual Learning\"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
import copy


class SDFTTrainer:
    \"\"\"
    Self-Distillation Fine-Tuning Trainer
    
    Implements the SDFT algorithm from the paper where a model acts as both student and teacher,
    with the teacher being the same model conditioned on expert demonstrations.
    \"\"\"
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        ema_alpha: float = 0.02,
        max_length: int = 2048
    ):
        \"\"\"
        Initialize the SDFT trainer
        
        Args:
            model_name: Name of the pre-trained model to use
            learning_rate: Learning rate for training
            ema_alpha: EMA decay rate for teacher model
            max_length: Maximum sequence length
        \"\"\"
        self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.student_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.teacher_model = copy.deepcopy(self.student_model)
        
        self.student_model.to(self.device)
        self.teacher_model.to(self.device)
        
        # Freeze teacher model initially
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        self.ema_alpha = ema_alpha
        self.max_length = max_length
        
    def create_teacher_context(self, query: str, demonstration: str) -> str:
        \"\"\"
        Create context for the teacher model according to the paper's prompt template
        
        <Question>
        This is an example for a response to the question:
        <Demonstration>
        Now answer with a response of your own, including the thinking process:
        \"\"\"
        context = (
            f\"<Question>\\n\"
            f\"{query}\\n\"
            f\"This is an example for a response to the question:\\n\"
            f\"<Demonstration>\\n\"
            f\"{demonstration}\\n\"
            f\"Now answer with a response of your own, including the thinking process:\"
        )
        return context
    
    def create_student_context(self, query: str) -> str:
        \"\"\"
        Create context for the student model (query only)
        \"\"\"
        return query
    
    def compute_reverse_kl_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        \"\"\"
        Compute the reverse KL divergence loss between student and teacher
        L(θ) = D_KL(π_θ(·|x) || π(·|x, c))
        
        Args:
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
            
        Returns:
            Scalar loss value
        \"\"\"
        # Convert logits to log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Calculate reverse KL: KL(P_student || P_teacher)
        # This is sum over vocab: P_student * log(P_student / P_teacher)
        kl_divergence = teacher_probs * (torch.log(teacher_probs + 1e-12) - student_log_probs)
        return kl_divergence.sum(dim=-1).mean()
    
    def get_model_responses(
        self, 
        queries: List[str], 
        demonstrations: List[str],
        max_new_tokens: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        Get responses from both student (on-policy) and teacher models
        
        Args:
            queries: List of input queries
            demonstrations: List of expert demonstrations
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Tuple of (student_hidden_states, teacher_hidden_states)
        \"\"\"
        # Prepare inputs for both student and teacher
        student_inputs = [
            self.create_student_context(query) for query in queries
        ]
        
        teacher_inputs = [
            self.create_teacher_context(query, demo) 
            for query, demo in zip(queries, demonstrations)
        ]
        
        # Tokenize inputs
        student_encoded = self.tokenizer(
            student_inputs,
            return_tensors=\"pt\",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        teacher_encoded = self.tokenizer(
            teacher_inputs,
            return_tensors=\"pt\",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate responses from student model (on-policy)
        with torch.no_grad():
            student_outputs = self.student_model.generate(
                **student_encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Get teacher outputs conditioned on demonstrations and queries
        teacher_outputs = self.teacher_model(**teacher_encoded)
        teacher_logits = teacher_outputs.logits
        
        # Get corresponding student logits for the same sequences
        student_outputs_truncated = student_outputs[:, :teacher_logits.size(1)]
        student_input_ids_for_logits = student_outputs_truncated
        student_outputs_full = self.student_model(student_input_ids_for_logits)
        student_logits = student_outputs_full.logits
        
        return student_logits, teacher_logits
    
    def update_teacher_model(self):
        \"\"\"
        Update teacher model using exponential moving average of student parameters
        \"\"\"
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(),
                self.student_model.parameters()
            ):
                teacher_param.data.mul_(1.0 - self.ema_alpha).add_(
                    student_param.data, alpha=self.ema_alpha
                )
    
    def train_step(
        self, 
        queries: List[str], 
        demonstrations: List[str]
    ) -> float:
        \"\"\"
        Perform one training step of SDFT
        
        Args:
            queries: List of input queries for the student
            demonstrations: List of expert demonstrations for the teacher
            
        Returns:
            Loss value for this step
        \"\"\"
        self.student_model.train()
        
        # Get logits from both models
        student_logits, teacher_logits = self.get_model_responses(queries, demonstrations)
        
        # Compute reverse KL divergence loss
        loss = self.compute_reverse_kl_loss(student_logits, teacher_logits)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update teacher model with EMA
        self.update_teacher_model()
        
        return loss.item()
    
    def train(
        self,
        train_data: List[Tuple[str, str]],  # List of (query, demonstration) pairs
        num_epochs: int = 3,
        batch_size: int = 4,
        eval_callback=None
    ):
        \"\"\"
        Train the model using SDFT
        
        Args:
            train_data: List of (query, demonstration) tuples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            eval_callback: Optional callback function for evaluation
        \"\"\"
        total_steps = 0
        
        for epoch in range(num_epochs):
            print(f\"Epoch {epoch + 1}/{num_epochs}\")
            
            # Shuffle training data
            import random
            shuffled_data = train_data.copy()
            random.shuffle(shuffled_data)
            
            # Process in batches
            for i in range(0, len(shuffled_data), batch_size):
                batch = shuffled_data[i:i + batch_size]
                
                queries = [item[0] for item in batch]
                demonstrations = [item[1] for item in batch]
                
                loss = self.train_step(queries, demonstrations)
                
                total_steps += 1
                
                if total_steps % 10 == 0:
                    print(f\"Step {total_steps}, Loss: {loss:.4f}\")
                    
                if eval_callback and total_steps % 50 == 0:
                    eval_result = eval_callback(self.student_model, self.tokenizer)
                    print(f\"Evaluation at step {total_steps}: {eval_result}\")
        
        print(\"Training completed!\")
    
    def save_model(self, output_path: str):
        \"\"\"
        Save the trained student model
        \"\"\"
        self.student_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)


def example_usage():
    \"\"\"
    Example usage of the SDFT trainer
    \"\"\"
    # Sample training data: (query, demonstration) pairs
    train_data = [
        (
            \"Solve the equation: 2x + 5 = 15\",
            \"To solve 2x + 5 = 15, subtract 5 from both sides: 2x = 10, then divide by 2: x = 5.\"
        ),
        (
            \"Explain photosynthesis\",
            \"Photosynthesis is the process where plants convert CO2 and water into glucose using sunlight. It occurs in chloroplasts and releases oxygen as a byproduct.\"
        ),
        (
            \"Write a function to calculate factorial\",
            \"def factorial(n):\\n    if n <= 1:\\n        return 1\\n    else:\\n        return n * factorial(n-1)\"
        ),
        (
            \"How does a computer execute a program?\",
            \"A computer executes a program by loading instructions into memory, fetching them from memory, decoding the instruction, executing it, and storing results. This fetch-decode-execute cycle repeats until the program ends.\"
        )
    ]
    
    # Initialize trainer (using a smaller model for demonstration)
    # In practice, you would use a larger model like Qwen2.5-7B-Instruct as mentioned in the paper
    trainer = SDFTTrainer(
        model_name=\"gpt2\",  # Using gpt2 for demonstration; in practice use larger models
        learning_rate=1e-5,
        ema_alpha=0.02
    )
    
    # Train the model
    trainer.train(
        train_data=train_data,
        num_epochs=2,
        batch_size=2
    )
    
    # Save the trained model
    trainer.save_model(\"./sdft_finetuned_model\")


if __name__ == \"__main__\":
    example_usage()