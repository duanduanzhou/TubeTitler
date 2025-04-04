#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_title_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("title_generator")

class TitleGenerationDataset(Dataset):
    def __init__(self, samples, tokenizer, clip_dim=512, max_transcript_length=512, max_title_length=64):

        self.samples = samples
        self.tokenizer = tokenizer
        self.clip_dim = clip_dim
        self.max_transcript_length = max_transcript_length
        self.max_title_length = max_title_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get transcript and title
        transcript = sample.get("transcript", "")
        title = sample.get("title", "")
        
        # Truncate or pad transcript and title
        input_encoding = self.tokenizer(
            transcript,
            max_length=self.max_transcript_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            title,
            max_length=self.max_title_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get embeddings and resize if needed
        original_clip_embedding = torch.tensor(sample.get("clip_embedding_avg", [0.0] * 512), dtype=torch.float)
        
        # Resize CLIP embedding if dimensions don't match
        if self.clip_dim != original_clip_embedding.shape[0]:
            if self.clip_dim < original_clip_embedding.shape[0]:
                # Truncate to smaller dimension
                clip_embedding = original_clip_embedding[:self.clip_dim]
            else:
                # Pad with zeros if needed
                clip_embedding = torch.zeros(self.clip_dim, dtype=torch.float)
                clip_embedding[:original_clip_embedding.shape[0]] = original_clip_embedding
        else:
            clip_embedding = original_clip_embedding
        
        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": target_encoding.input_ids.squeeze(),
            "clip_embedding": clip_embedding
        }

class MultimodalTitleGenerator(nn.Module):
    # Multimodal title generation model
    
    def __init__(self, text_model_name="t5-base", clip_dim=512, fusion_dim=512):
        super().__init__()
        
        # Text generator (T5)
        self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.text_model_config = self.text_model.config
        
        # Projection for CLIP embeddings
        self.clip_projection = nn.Sequential(
            nn.Linear(clip_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # Fusion mechanism
        self.fusion_layer = nn.Linear(fusion_dim, self.text_model_config.d_model)
        
    def forward(self, input_ids, attention_mask, clip_embedding, labels=None):
        """Forward pass with multimodal fusion."""
        
        # Project CLIP embeddings
        projected_clip = self.clip_projection(clip_embedding)
        
        # Fusion
        fusion_vector = self.fusion_layer(projected_clip)
        
        # T5 encoder outputs
        encoder_outputs = self.text_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Inject visual features
        hidden_states = encoder_outputs.last_hidden_state
        
        # Add fusion vector to each token representation
        fusion_vector = fusion_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]
        enhanced_hidden_states = hidden_states + fusion_vector
        
        # T5 decoder
        if labels is not None:
            # Training mode
            outputs = self.text_model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            # Inference mode
            outputs = self.text_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=64
            )
            return outputs
            
    def generate(self, input_ids, attention_mask, clip_embedding, **generation_kwargs):
        """Generate title sequences (for inference)"""
        # Project CLIP embeddings
        projected_clip = self.clip_projection(clip_embedding)
        
        # Fusion
        fusion_vector = self.fusion_layer(projected_clip)
        
        # T5 encoder outputs
        encoder_outputs = self.text_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Inject visual features into the encoder hidden states
        hidden_states = encoder_outputs.last_hidden_state
        
        # Add fusion vector to each token representation
        fusion_vector = fusion_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]
        enhanced_hidden_states = hidden_states + fusion_vector
        
        # Generate sequences using the model's generate method
        outputs = self.text_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        return outputs

def load_dataset(dataset_path):
    """Load dataset from JSON or CSV file."""
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    elif dataset_path.endswith('.csv'):
        data = pd.read_csv(dataset_path).to_dict('records')
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    return data

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, output_dir, num_epochs=3):
    """Train the title generation model."""
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            clip_embedding = batch["clip_embedding"].to(device)
            
            optimizer.zero_grad()
            
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_embedding=clip_embedding,
                labels=labels
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                clip_embedding = batch["clip_embedding"].to(device)
                
                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    clip_embedding=clip_embedding,
                    labels=labels
                )
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save model state dictionary
            model_path = os.path.join(output_dir, f"title_generator_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
    
    # Save the complete model and tokenizer structure
    try:
        # Save the T5 model and tokenizer
        model.text_model.save_pretrained(output_dir)
        # Save config as json
        model_config = {
            "clip_dim": model.clip_projection[0].weight.size(1),
            "fusion_dim": model.clip_projection[0].weight.size(0),
            "model_type": "t5"
        }
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Saved complete model and config to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving complete model: {str(e)}")
    
    return model

def generate_titles(model, tokenizer, samples, device):
    # Generate titles for test samples
    model.to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Generating titles"):
            transcript = sample.get("transcript", "")
            
            # Tokenize transcript
            inputs = tokenizer(
                transcript,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Get CLIP embedding
            original_clip_embedding = torch.tensor(
                sample.get("clip_embedding_avg", [0.0] * 512), 
                dtype=torch.float
            )
            
            # Resize CLIP embedding to match the model's expected dimension
            clip_dim = model.clip_projection[0].weight.size(1)  # get clip_dim from model
            if clip_dim != original_clip_embedding.shape[0]:
                if clip_dim < original_clip_embedding.shape[0]:
                    # Truncate to smaller dimension
                    clip_embedding = original_clip_embedding[:clip_dim]
                else:
                    # Pad with zeros if needed
                    clip_embedding = torch.zeros(clip_dim, dtype=torch.float)
                    clip_embedding[:original_clip_embedding.shape[0]] = original_clip_embedding
            else:
                clip_embedding = original_clip_embedding
                
            clip_embedding = clip_embedding.unsqueeze(0).to(device)
            
            # Generate title
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_embedding=clip_embedding,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode title
            generated_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                "video_id": sample.get("video_id", ""),
                "original_title": sample.get("title", ""),
                "generated_title": generated_title
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train title generation model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file (JSON or CSV)")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Base model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    parser.add_argument("--clip_dim", type=int, default=128, help="CLIP embedding dimension (reduced from 512)")
    parser.add_argument("--fusion_dim", type=int, default=64, help="Fusion layer dimension")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    data = load_dataset(args.dataset)
    logger.info(f"Loaded {len(data)} samples")
    
    # Split dataset
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    logger.info(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = TitleGenerationDataset(train_data, tokenizer, clip_dim=args.clip_dim)
    val_dataset = TitleGenerationDataset(val_data, tokenizer, clip_dim=args.clip_dim)
    test_dataset = TitleGenerationDataset(test_data, tokenizer, clip_dim=args.clip_dim)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model with smaller dimensions
    model = MultimodalTitleGenerator(
        text_model_name=args.model_name,
        clip_dim=args.clip_dim,
        fusion_dim=args.fusion_dim
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Train model
    logger.info("Starting training")
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        args.output_dir,
        num_epochs=args.epochs
    )
    
    # Save tokenizer
    try:
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved tokenizer to {args.output_dir}")
    except Exception as e:
        logger.error(f"Error saving tokenizer: {str(e)}")
    
    # Generate titles for test set
    logger.info("Generating titles for test set")
    results = generate_titles(model, tokenizer, test_data, device)
    
    output_file = os.path.join(args.output_dir, "title_generation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 